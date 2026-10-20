"""Weight-free CUDA smoke: real native MOSS blocks and GPU/CPU cache transfers."""

import hashlib
import json
import os
from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.environ["COMFYUI_PATH"])
package = types.ModuleType("moss_blockcache_t8")
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package

from comfy.cli_args import args
args.use_pytorch_cross_attention = True

import torch
import comfy.ops
from comfy.ldm.moss_image.model import MOSSImage
from comfy.patcher_extension import WrapperExecutor
from moss_blockcache_t8.cache import CacheConfig, CacheRuntime
from moss_blockcache_t8.runtime import RUNTIME, diffusion_wrapper


class Sampling:
    def percent_to_sigma(self, p):
        return 1 - p


def run(native, cache_device, threshold):
    runtime = CacheRuntime(CacheConfig(cache_device=cache_device, threshold=threshold, metric_stride=1), Sampling())
    calls = [0] * len(native.layers)
    head_calls = []
    handles = [native.final_linear_video.register_forward_hook(lambda *unused: head_calls.append(1))]
    for index, layer in enumerate(native.layers):
        def count(module, inputs, output, i=index):
            calls[i] += 1
        handles.append(layer.register_forward_hook(count))
    executor = WrapperExecutor.new_class_executor(native._forward, native, [diffusion_wrapper])
    x = torch.ones(1, 8, 2, 2, device="cuda", dtype=torch.bfloat16)
    context = torch.ones(1, 3, 16, device="cuda", dtype=torch.bfloat16)
    timestep = torch.tensor([0.5], device="cuda", dtype=torch.bfloat16)
    outputs = []
    try:
        with torch.inference_mode():
            for sigma in (0.8, 0.7, 0.6):
                options = {RUNTIME: runtime, "uuids": ["positive"], "cond_or_uncond": [0],
                           "sigmas": torch.tensor([sigma], device="cuda")}
                outputs.append(executor.execute(x, timestep, context, options).clone())
        torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    expected = [3, 2, 2, 2] if threshold else [3, 3, 3, 3]
    assert calls == expected and len(head_calls) == 3
    assert runtime.hits == (1 if threshold else 0)
    assert all(bool(torch.isfinite(out).all()) for out in outputs)
    if threshold:
        stream = next(iter(runtime.streams.values()))
        assert stream.tail.device.type == ("cpu" if cache_device == "cpu" else "cuda")
    report = {"cache_device": cache_device, "threshold": threshold, "layer_calls": calls,
              "head_calls": len(head_calls), "hits": runtime.hits, "skipped_blocks": runtime.skipped_blocks,
              "retained_bytes": runtime.bytes()}
    runtime.clear()
    assert runtime.bytes() == 0
    return outputs, report


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA verification requires a CUDA device")
    torch.manual_seed(42)
    native = MOSSImage(dim=128, ffn_dim=256, num_heads=1, num_layers=4, text_dim=16,
                       latent_channels=8, dtype=torch.bfloat16, device="cuda", operations=comfy.ops.disable_weight_init)
    with torch.no_grad():
        for parameter in native.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.02)
    gpu, gpu_report = run(native, "gpu", 0.05)
    cpu, cpu_report = run(native, "cpu", 0.05)
    zero, zero_report = run(native, "gpu", 0)
    assert all(torch.equal(a, b) for a, b in zip(gpu, cpu))
    assert torch.equal(gpu[0], zero[0]) and torch.equal(gpu[2], zero[2])
    report = {"status": "passed", "scope": "Miniature real native MOSS, BF16 CUDA; not full-weight image quality or performance.",
              "device": torch.cuda.get_device_name(), "torch": torch.__version__,
              "source_sha256": {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest()
                                for n in ("cache.py", "runtime.py", "nodes.py")},
              "gpu_cpu_outputs_pixel_equal": True, "cases": [gpu_report, cpu_report, zero_report]}
    destination = ROOT / "artifacts/gpu-smoke.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
