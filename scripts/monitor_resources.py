"""Sample the dedicated verifier process tree and physical GPU usage, locally."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import psutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    owner = psutil.Process(args.pid)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Monitoring owned verifier PID {args.pid}", flush=True)
    with args.output.open("a", encoding="utf-8", buffering=1) as log:
        while owner.is_running():
            processes = [owner] + owner.children(recursive=True)
            rss = sum(v.memory_info().rss for v in processes if v.is_running())
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10, creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
            gpu = [[float(n.strip()) for n in line.split(",")] for line in out.stdout.strip().splitlines()]
            log.write(json.dumps({"epoch":time.time(), "rss_bytes":rss, "gpu_memory_mib_and_utilization":gpu})+"\n")
            time.sleep(2)


if __name__ == "__main__":
    main()
