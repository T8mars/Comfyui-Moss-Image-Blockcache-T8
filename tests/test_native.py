import importlib.util
from types import SimpleNamespace

import pytest
import torch

if importlib.util.find_spec("comfy") is None:
    pytest.skip("Set COMFYUI_PATH to run native integration tests", allow_module_level=True)

import comfy.model_patcher
import comfy.ops
from comfy.ldm.moss_image.model import MOSSImage
from comfy.patcher_extension import WrapperExecutor, WrappersMP

from moss_blockcache_t8.cache import CacheConfig, CacheRuntime
from moss_blockcache_t8.runtime import FRAME, KEY, RUNTIME, diffusion_wrapper, install, sample_wrapper


@pytest.fixture
def native():
    torch.manual_seed(42)
    m = MOSSImage(dim=128, ffn_dim=256, num_heads=1, num_layers=4, text_dim=16,
                  latent_channels=8, dtype=torch.float32, device="cpu", operations=comfy.ops.disable_weight_init)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn_like(p) * 0.02)
    return m


class Sampling:
    def percent_to_sigma(self, p):
        return 1 - p


def invoke(native, runtime, sigma, *, identity="positive", text_len=3, extra=None, x=None):
    options = {RUNTIME: runtime, "uuids": [identity], "cond_or_uncond": [0 if identity == "positive" else 1],
               "sigmas": torch.tensor([sigma])}
    options.update(extra or {})
    x = torch.ones(1, 8, 2, 2) if x is None else x
    context = torch.ones(x.shape[0], text_len, 16)
    executor = WrapperExecutor.new_class_executor(native._forward, native, [diffusion_wrapper])
    # Fixed timestep makes the test measure residual reuse, independently of a learned approximation.
    return executor.execute(x, torch.ones(x.shape[0]) * 0.5, context, options)


def test_no_reuse_matches_real_native_and_does_not_modify_options(native):
    r = CacheRuntime(CacheConfig(threshold=0), Sampling())
    with torch.inference_mode():
        actual = invoke(native, r, 0.8)
        expected = native(torch.ones(1, 8, 2, 2), torch.tensor([0.5]), torch.ones(1, 3, 16))
    assert torch.equal(actual, expected)
    assert r.hits == 0


def test_enabled_guard_with_no_hits_is_exactly_native(native):
    r = CacheRuntime(CacheConfig(threshold=1e-12, metric_stride=1), Sampling())
    with torch.inference_mode():
        for sigma, scale in ((0.8, 1.0), (0.7, 1.2), (0.6, 0.8)):
            x = torch.ones(1, 8, 2, 2) * scale
            actual = invoke(native, r, sigma, x=x)
            expected = native(x, torch.tensor([0.5]), torch.ones(1, 3, 16))
            assert torch.equal(actual, expected)
    assert r.hits == 0 and r.full == 3
    assert next(iter(r.streams.values())).tail is not None


def test_inplace_native_reuse_actually_skips_layers(native):
    r = CacheRuntime(CacheConfig(threshold=0.1, metric_stride=1), Sampling())
    calls = [0] * len(native.layers)
    handles = []
    head_calls = []
    handles.append(native.final_linear_video.register_forward_hook(lambda *args: head_calls.append(1)))
    for index, layer in enumerate(native.layers):
        def count(module, args, output, i=index):
            calls[i] += 1
        handles.append(layer.register_forward_hook(count))
    with torch.inference_mode():
        first = invoke(native, r, 0.8)
        saved = next(iter(r.streams.values())).tail.clone()
        second = invoke(native, r, 0.7)
        third = invoke(native, r, 0.6)
    for h in handles:
        h.remove()
    assert calls == [3, 2, 2, 2]
    assert len(head_calls) == 3
    assert r.hits == 1 and r.skipped_blocks == 3
    assert torch.allclose(first, second, atol=1e-7, rtol=1e-6)
    assert torch.equal(first, third)
    assert torch.equal(saved, next(iter(r.streams.values())).tail)


def test_two_stage_window_with_real_native_blocks(native):
    r = CacheRuntime(CacheConfig(threshold=0, threshold_mode="two_stage", late_threshold=0.05,
                                 split_ratio=0.5, metric_stride=1), Sampling())
    with torch.inference_mode():
        outputs = [invoke(native, r, sigma) for sigma in (0.8, 0.49, 0.48, 0.05)]
    assert r.full == 3 and r.hits == 1
    assert r.skipped_blocks == 3
    assert all(torch.allclose(outputs[0], out, atol=1e-7, rtol=1e-6) for out in outputs[1:])


def test_external_negative_patch_does_not_destroy_positive_history(native):
    r = CacheRuntime(CacheConfig(metric_stride=1), Sampling())
    def skip(args, extra):
        return args
    with torch.inference_mode():
        invoke(native, r, 0.8)
        invoke(native, r, 0.8, identity="negative", text_len=1,
               extra={"patches_replace": {"dit": {("single_block", 2): skip}}})
        invoke(native, r, 0.7)
    assert r.hits == 1
    assert r.fallbacks == {"external_block_patch": 1}


def test_mixed_branch_patch_invalidates_overlapping_history(native):
    r = CacheRuntime(CacheConfig(metric_stride=1), Sampling())
    def skip(args, extra):
        return args
    with torch.inference_mode():
        invoke(native, r, 0.8)
        invoke(native, r, 0.7, x=torch.ones(2, 8, 2, 2), extra={
            "uuids": ["positive", "negative"], "cond_or_uncond": [0, 1],
            "patches_replace": {"dit": {("single_block", 2): skip}},
        })
        invoke(native, r, 0.6)
    assert r.hits == 0


def test_nonuniform_sigma_clears_previous_history(native):
    r = CacheRuntime(CacheConfig(metric_stride=1), Sampling())
    with torch.inference_mode():
        invoke(native, r, 0.8)
        invoke(native, r, 0.7, extra={"sigmas": torch.tensor([0.7, 0.6])}, x=torch.ones(2, 8, 2, 2))
        invoke(native, r, 0.6)
    assert r.hits == 0
    assert r.fallbacks == {"nonuniform_sigma": 1}


@pytest.mark.parametrize("extra", [
    {"uuids": None}, {"uuids": []}, {"uuids": [None]}, {"cond_or_uncond": None}, {"cond_or_uncond": []},
])
def test_missing_condition_identity_uses_native_forward(native, extra):
    r = CacheRuntime(CacheConfig(metric_stride=1), Sampling())
    with torch.inference_mode():
        expected = native(torch.ones(1, 8, 2, 2), torch.tensor([0.5]), torch.ones(1, 3, 16))
        invoke(native, r, 0.8)
        actual = invoke(native, r, 0.7, extra=extra)
        invoke(native, r, 0.6)
    assert torch.equal(actual, expected)
    assert r.hits == 0
    assert r.fallbacks == {"missing_condition_identity": 1}


@pytest.mark.parametrize("change", ["identity", "text", "shape", "rope", "repeat", "increase"])
def test_condition_shape_rope_and_time_invalidate(native, change):
    r = CacheRuntime(CacheConfig(metric_stride=1), Sampling())
    with torch.inference_mode():
        invoke(native, r, 0.8)
        args = {"identity": "other"} if change == "identity" else {}
        if change == "text":
            args["text_len"] = 4
        if change == "shape":
            args["x"] = torch.ones(2, 8, 2, 2)
        if change == "rope":
            args["extra"] = {"rope_options": {"scale_x": 1.2}}
        invoke(native, r, 0.8 if change == "repeat" else 0.9 if change == "increase" else 0.7, **args)
    assert r.hits == 0


@pytest.mark.parametrize("error", [None, RuntimeError, KeyboardInterrupt])
def test_sampling_runtime_is_local_and_cleared_even_on_cancel(error):
    config = CacheConfig()
    original = {"transformer_options": {KEY: config}}
    patcher = SimpleNamespace(model=SimpleNamespace(model_sampling=Sampling()), hook_patches={}, object_patches={})
    guider = SimpleNamespace(model_options=original, model_patcher=patcher, conds={"positive": []})
    captured = []
    def sample():
        runtime = guider.model_options["transformer_options"][RUNTIME]
        captured.append(runtime)
        runtime.stream("test", 0.8).tail = torch.ones(2)
        if error:
            raise error("interrupted")
        return 123
    executor = WrapperExecutor.new_class_executor(sample, guider, [sample_wrapper])
    if error:
        with pytest.raises(error):
            executor.execute()
    else:
        assert executor.execute() == 123
        assert executor.execute() == 123
        assert captured[0] is not captured[1]
    assert guider.model_options is original
    assert all(r.bytes() == 0 for r in captured)
    assert RUNTIME not in original["transformer_options"]


@pytest.mark.parametrize("object_key", ["diffusion_model", "diffusion_model.layers.0"])
def test_entire_model_and_nested_object_replacements_use_full_computation(object_key):
    options = {"transformer_options": {KEY: CacheConfig()}}
    patcher = SimpleNamespace(model=SimpleNamespace(model_sampling=Sampling()),
                              hook_patches={}, object_patches={object_key: object()})
    guider = SimpleNamespace(model_options=options, model_patcher=patcher, conds={"positive": []})
    def sample():
        return guider.model_options["transformer_options"].get(FRAME)
    executor = WrapperExecutor.new_class_executor(sample, guider, [sample_wrapper])
    assert executor.execute() == "external_model_object_patch"
    assert guider.model_options is options


@pytest.mark.parametrize("reason", [
    "unsupported_conditioning", "external_model_wrapper", "external_model_object_patch",
])
def test_execution_restriction_keeps_native_result_and_invalidates_history(native, reason):
    r = CacheRuntime(CacheConfig(metric_stride=1), Sampling())
    with torch.inference_mode():
        invoke(native, r, 0.8)
        x = torch.ones(1, 8, 2, 2) * 1.2
        actual = invoke(native, r, 0.7, x=x, extra={FRAME: reason})
        expected = native(x, torch.tensor([0.5]), torch.ones(1, 3, 16))
        invoke(native, r, 0.6)
    assert torch.equal(actual, expected)
    assert r.hits == 0 and r.fallbacks == {reason: 1}


def test_install_is_clone_local_and_idempotent(native):
    base = torch.nn.Module()
    base.diffusion_model = native
    base.model_dtype = lambda: torch.float32
    patcher = comfy.model_patcher.ModelPatcher(base, torch.device("cpu"), torch.device("cpu"))
    cached = install(patcher, CacheConfig())
    again = install(cached, CacheConfig(threshold=0.1))
    assert KEY not in patcher.model_options["transformer_options"]
    assert len(again.get_wrappers(WrappersMP.OUTER_SAMPLE, KEY)) == 1
    assert len(again.get_wrappers(WrappersMP.DIFFUSION_MODEL, KEY)) == 1
    disabled = install(cached, CacheConfig(), enabled=False)
    assert not disabled.get_wrappers(WrappersMP.DIFFUSION_MODEL, KEY)
    assert KEY not in disabled.model_options["transformer_options"]
    zero = install(cached, CacheConfig(threshold=0))
    assert not zero.get_wrappers(WrappersMP.OUTER_SAMPLE, KEY)
    assert not zero.get_wrappers(WrappersMP.DIFFUSION_MODEL, KEY)
    assert KEY not in zero.model_options["transformer_options"]

