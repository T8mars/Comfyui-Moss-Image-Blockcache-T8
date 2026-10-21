"""Resumable paired quality matrix; run on a dedicated, idle, offline ComfyUI server."""

import argparse
import hashlib
import json
from pathlib import Path

from verify_native import PROMPTS, make_prompt, run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8291")
    p.add_argument("--output", type=Path, default=Path("artifacts/matrix"))
    p.add_argument("--log", type=Path, default=Path("artifacts/server-8291-error.log"))
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--cases", default=",".join(PROMPTS))
    p.add_argument("--variants", default="sft,base")
    p.add_argument("--threshold", type=float, default=0.08)
    p.add_argument("--deployment", type=Path)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = {n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ("cache.py", "runtime.py", "nodes.py")}
    if args.deployment:
        deployed = json.loads(args.deployment.read_text(encoding="utf-8-sig"))
        if deployed.get("source_sha256") != source:
            raise RuntimeError("The dedicated server was not started with the current source")
    args.output.mkdir(parents=True, exist_ok=True)
    done = 0
    for variant in args.variants.split(","):
        for case in args.cases.split(","):
            for seed in map(int, args.seeds.split(",")):
                for mode in ("baseline", "cache"):
                    destination = args.output / f"{variant}-{case}-{seed}-{mode}.json"
                    graph = make_prompt(variant, case, seed, args.size, args.steps, mode, args.threshold, tag="matrix")
                    if destination.exists():
                        old = json.loads(destination.read_text(encoding="utf-8"))
                        if old.get("source_sha256") == source and old.get("prompt") == graph:
                            done += 1
                            continue
                        raise RuntimeError(f"Existing result does not match current source/parameters: {destination}")
                    report = run(args.url, graph, destination, args.log)
                    if mode == "cache" and (not report["cache"] or report["cache"][-1]["hits"] == 0):
                        raise RuntimeError(f"No real cache hits: {destination}")
                    done += 1
                    state = {"completed": done, "last": str(destination), "source_sha256": source,
                             "size": args.size, "steps": args.steps}
                    (args.output / "progress.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
                    print("MATRIX", json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
