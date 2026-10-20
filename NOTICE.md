# Sources and attribution

Copyright 2026 T8mars. Licensed under Apache-2.0; see LICENSE.

The residual-cache mathematics and execution-scoped wrapper pattern adapt the author's
[Qwen-Image-2.1 BlockCache T8](https://github.com/T8mars/Comfyui-Qwen-Image-2.1-BlockCache-T8),
revision `53c9d3fbce4853c543a6c51613baa4babf0c15b5`, Apache-2.0.
The MOSS adapter uses native single-block replacement hooks and preserves the original
loop, prefetch cleanup and output head. No upstream speed or quality claim is inherited.

[MOSS-Image](https://github.com/OpenMOSS/MOSS-Image) and its models are by OpenMOSS.
The plugin targets [ComfyUI](https://github.com/Comfy-Org/ComfyUI)'s native implementation;
reference revision `8f04fd6ecf4a548937971e840af2a985529fab76`.
No ComfyUI model implementation, model weights, tokenizer files or GPU kernels are bundled.
Model licenses and attribution remain with their respective owners.
