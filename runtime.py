from dataclasses import asdict
import json
import logging
import math

import torch
import comfy.model_patcher
import comfy.model_prefetch as prefetch
import comfy.patcher_extension as patches
from comfy.ldm.moss_image.model import MOSSImage

from .cache import CacheRuntime, ForwardState


KEY = "moss_image_blockcache_t8"
RUNTIME = KEY + "_runtime"
FRAME = KEY + "_frame"
log = logging.getLogger(__name__)


def sample_wrapper(executor, *args, **kwargs):
    guider = executor.class_obj
    original = guider.model_options
    local = comfy.model_patcher.create_model_options_clone(original)
    options = local["transformer_options"]
    runtime = CacheRuntime(options[KEY], guider.model_patcher.get_model_object("model_sampling"))
    options[RUNTIME] = runtime
    # Conditions with a changing spatial or weight meaning need their original execution.
    unsafe = {"area", "mask", "control", "gligen", "hooks"}
    denoise_mask = kwargs.get("denoise_mask", args[4] if len(args) > 4 else None)
    restricted = denoise_mask is not None or any(unsafe.intersection(c) for cond in guider.conds.values() for c in cond)
    if restricted or guider.model_patcher.hook_patches:
        options[FRAME] = "unsupported_conditioning"
    elif local.get("model_function_wrapper") is not None or patches.get_all_wrappers(patches.WrappersMP.APPLY_MODEL, options):
        options[FRAME] = "external_model_wrapper"
    elif any(k == "diffusion_model" or k.startswith("diffusion_model.") for k in guider.model_patcher.object_patches):
        options[FRAME] = "external_model_object_patch"
    guider.model_options = local
    try:
        return executor(*args, **kwargs)
    finally:
        summary = {"full": runtime.full, "hits": runtime.hits, "skipped_blocks": runtime.skipped_blocks,
                   "peak_cache_mib": round(runtime.peak_bytes / 1048576, 3), "fallbacks": runtime.fallbacks,
                   "config": asdict(runtime.config)}
        runtime.clear()
        guider.model_options = original
        log.info("MOSS BlockCache T8: %s", json.dumps(summary, sort_keys=True))


def unsupported(options, kwargs):
    reason = options.get(FRAME)
    if isinstance(reason, str):
        return reason
    if options.get("patches_replace", {}).get("dit"):
        return "external_block_patch"
    if any(options.get("patches", {}).values()):
        return "external_patch"
    if "easycache" in options:
        return "external_cache"
    if any(w is not diffusion_wrapper for w in patches.get_all_wrappers(patches.WrappersMP.DIFFUSION_MODEL, options)):
        return "external_diffusion_wrapper"
    if kwargs.get("control") is not None:
        return "control"
    uuids = options.get("uuids")
    branches = options.get("cond_or_uncond")
    if not uuids or not branches or len(uuids) != len(branches) or any(u is None for u in uuids):
        return "missing_condition_identity"
    return None


def diffusion_wrapper(executor, x, timestep, context, transformer_options=None, **kwargs):
    options = dict(transformer_options or {})
    runtime = options.get(RUNTIME)
    if runtime is None:
        return executor(x, timestep, context, options, **kwargs)
    reason = unsupported(options, kwargs)
    sigmas = options.get("sigmas")
    sigma = float(sigmas.flatten()[0]) if isinstance(sigmas, torch.Tensor) and sigmas.numel() else None
    if sigma is None or not math.isfinite(sigma) or not bool((sigmas == sigma).all()):
        reason = reason or "nonuniform_sigma"
    runtime.full += 1
    if reason:
        runtime.fallback(reason)
        identity = (tuple(map(str, options.get("uuids") or [])), tuple(options.get("cond_or_uncond") or []))
        if reason in ("nonuniform_sigma", "missing_condition_identity"):
            runtime.clear()
        elif identity[0]:
            affected = set(zip(*identity))
            for key in list(runtime.streams):
                if affected.intersection(zip(key[0], key[1])):
                    runtime.streams.pop(key)
        else:
            runtime.clear()
        return executor(x, timestep, context, options, **kwargs)
    rope = options.get("rope_options", {})
    key = (tuple(map(str, options["uuids"])), tuple(options["cond_or_uncond"]),
           tuple(x.shape), tuple(context.shape), x.dtype, x.device, tuple(sorted(rope.items())))
    stream = runtime.stream(key, sigma)
    if not runtime.active(sigma):
        stream.clear()
        return executor(x, timestep, context, options, **kwargs)
    count = len(executor.class_obj.layers)
    frame = ForwardState(runtime, stream, sigma, x.shape[-2] * x.shape[-1], count)
    options[FRAME] = frame
    replacements = {("single_block", i): block_patch for i in range(count)}
    options["patches_replace"] = {**options.get("patches_replace", {}), "dit": replacements}
    try:
        return executor(x, timestep, context, options, **kwargs)
    finally:
        frame.anchor = None
        frame.indicator = None
        frame.replay = None


def block_patch(args, extra_args):
    options = args["transformer_options"]
    frame = options[FRAME]
    runtime = frame.runtime
    index = options["block_index"]
    original = extra_args["original_block"]
    if index == 0:
        with prefetch.pause_malloc_graph():
            before = runtime.sample(args["img"], frame.image_tokens)
        out = original(args)["img"]
        with prefetch.pause_malloc_graph():
            after = runtime.sample(out, frame.image_tokens)
            frame.indicator = tuple(a - b for a, b in zip(after, before))
            frame.replay = runtime.replay(frame.stream, frame.indicator, frame.sigma, out)
            if frame.replay is not None:
                runtime.full -= 1
                runtime.skipped_blocks += frame.block_count - 1
            elif out.numel() * out.element_size() <= runtime.budget:
                frame.anchor = out.detach().clone()
            else:
                frame.stream.clear()
                runtime.fallback("anchor_budget")
        return {"img": out}
    if frame.replay is not None:
        return {"img": frame.replay if index == frame.block_count - 1 else args["img"]}
    out = original(args)["img"]
    if index == frame.block_count - 1 and frame.anchor is not None:
        with prefetch.pause_malloc_graph():
            runtime.store(frame.stream, frame.indicator, out, frame.anchor)
        frame.anchor = None
    return {"img": out}


def install(model, config, enabled=True):
    if not isinstance(model.model.diffusion_model, MOSSImage):
        raise ValueError("MOSS BlockCache T8 requires ComfyUI's native MOSS-Image MODEL")
    cloned = model.clone()
    for kind in (patches.WrappersMP.OUTER_SAMPLE, patches.WrappersMP.DIFFUSION_MODEL):
        cloned.remove_wrappers_with_key(kind, KEY)
    options = cloned.model_options["transformer_options"].copy()
    options.pop(KEY, None)
    if enabled and (config.threshold > 0 or config.threshold_mode == "two_stage" and config.late_threshold > 0):
        options[KEY] = config
        cloned.add_wrapper_with_key(patches.WrappersMP.OUTER_SAMPLE, KEY, sample_wrapper)
        cloned.add_wrapper_with_key(patches.WrappersMP.DIFFUSION_MODEL, KEY, diffusion_wrapper)
    cloned.model_options["transformer_options"] = options
    return cloned

