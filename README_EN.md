# ComfyUI MOSS Image BlockCache T8

[简体中文](README.md) · **English**

Experimental BlockCache acceleration for **ComfyUI's native MOSS-Image Base / SFT**. First-block change detection and tail-residual reuse reduce repeated computation while preserving native model loading, sampling and VAE decoding. Uses existing local models with no additional inference dependencies.

[Comfy Registry](https://registry.comfy.org/publishers/t8star/nodes/moss-image-blockcache-t8) · [Benchmarks](BENCHMARKS.md) · [Issues](https://github.com/T8mars/Comfyui-Moss-Image-Blockcache-T8/issues)

## Installation

Requires **ComfyUI 0.42.0 or newer with native MOSS-Image support**.

Search for **MOSS Image BlockCache T8** in ComfyUI Manager (package ID: `moss-image-blockcache-t8`), install and restart. Alternatively, run this inside `ComfyUI/custom_nodes`:

```bash
git clone https://github.com/T8mars/Comfyui-Moss-Image-Blockcache-T8.git comfyui-moss-image-blockcache-T8
```

The plugin does not download, convert or modify models. If custom nodes are restricted by a whitelist, add the installation directory name to `--whitelist-custom-nodes`.

## Workflows

Both Base and SFT were run through the real ComfyUI canvas. The saved JSON and generated PNG workflows were imported back for verification.

| Model | Canvas workflow | Image with embedded workflow |
|---|---|---|
| Base | [JSON](workflows/MOSS_T8_BASE_1024_BlockCache.json) | [PNG](workflows/MOSS_T8_BASE_1024_BlockCache.png) |
| SFT | [JSON](workflows/MOSS_T8_SFT_1024_BlockCache.json) | [PNG](workflows/MOSS_T8_SFT_1024_BlockCache.png) |

Drop a JSON or PNG onto the canvas. Select your native DiT, text encoder with `type=moss_image`, and matching VAE. Examples preselect T8 BF16 filenames; change these to your local files. The `*-api.json` files use the API format.

```text
Load Diffusion Model → MOSS Image Block Cache (T8)
    → SkipLayerGuidanceDiTSimple (negative-branch skip22) → KSampler
```

Find the node under `T8/MOSS-Image`. Its input and output use the standard `MODEL` type. Examples use 1024×1024, 50 Euler / simple steps, CFG 3 and seed 42.

**Keep MOSS's native shift=5 and timestep multiplier=1. Do not add the generic `ModelSamplingSD3` node: it changes the timestep scale and produces incorrect results.**

## Controls

| Parameter | Default and behavior |
|---|---|
| `enabled` | Disable to pass the model through for a native baseline |
| `threshold` | `0.05`; higher values allow more reuse and may change text, faces and details |
| `start_percent` / `end_percent` | `0.10 / 0.90`; compute fully outside this interval |
| `max_consecutive_hits` | `1`; force a refresh after a hit |
| `cache_device` | `gpu`; `cpu` trades transfer overhead for lower retained GPU cache usage |
| `max_cache_mb` / `metric_stride` | `512 MiB / 4`; retained-cache budget and metric subsampling interval |
| `threshold_mode` | `constant`; optional `two_stage` uses `split_ratio` and `late_threshold` |

In constant mode, `threshold=0` disables reuse. In two-stage mode, also set `late_threshold=0`, or disable `enabled`. The cache budget is not a total workflow VRAM limit.

## Validation and compatibility

Validated with native dense BF16 Base/SFT text-to-image and JSON Layout. Full-weight SFT zero-threshold output matched the native baseline pixel for pixel. Conditions, finite-sample timings and reviewed differences are documented in [BENCHMARKS.md](BENCHMARKS.md).

`0.1.2` fixes extreme metric overflow and cache invalidation with combined fallback conditions, and bounds empty streams. Canvas and timing records cover `0.1.0`; this patch uses native CPU regression checks. See the [review record](validation/review-0.1.2.json).

Caching is approximate and may change faces, text, hands, layout and details. Unsupported combinations such as spatial conditioning, Control, scheduled weight hooks and external block patches use full computation. Native negative-branch skip22 is preserved; the positive branch can still benefit. Do not stack other cache plugins. Quantized weights, reference-image editing, compilation, low-memory offload, other GPUs and other dtypes are outside this release's measured scope.

## Community and resources

[Bilibili](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [Hugging Face](https://huggingface.co/t8star)

[API](https://api.seedance.nz/sign-up?aff=5f4w) · [Free gallery](https://www.openzhenzhen.com) · [Online AI apps](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI bundle](https://pan.quark.cn/s/264edb7e36bd)

## License and credits

[Apache-2.0](LICENSE). Adapted from [Qwen-Image-2.1 BlockCache T8](https://github.com/T8mars/Comfyui-Qwen-Image-2.1-BlockCache-T8) for [MOSS-Image](https://github.com/OpenMOSS/MOSS-Image) and [ComfyUI](https://github.com/Comfy-Org/ComfyUI)'s native implementation. Model weights are not distributed with the plugin. See [NOTICE](NOTICE.md).
