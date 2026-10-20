import os
from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType("moss_blockcache_t8")
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package

comfy_root = os.environ.get("COMFYUI_PATH")
if comfy_root:
    sys.path.insert(0, comfy_root)
    from comfy.cli_args import args
    args.cpu = True
    args.use_pytorch_cross_attention = True

