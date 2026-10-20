"""Same-seed timing/noise checks after the quality matrix; never run concurrently."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

from PIL import Image

from report_matrix import image_path, metrics
from verify_native import make_prompt, run


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8291")
    parser.add_argument("--log", type=Path, default=ROOT / "artifacts/server-8291-error.log")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/repeats")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True)
    source = {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ("cache.py", "runtime.py", "nodes.py")}
    deployment = json.loads((ROOT / "artifacts/server-deployment.json").read_text(encoding="utf-8-sig"))
    if deployment["source_sha256"] != source:
        raise ValueError("Restart the dedicated verifier with the current source first")
    quality = json.loads((ROOT / "artifacts/matrix/progress.json").read_text(encoding="utf-8"))
    if quality.get("completed") != 72 or quality.get("source_sha256") != source:
        raise ValueError("Finish the current-source quality matrix before repeat timing")
    groups = []
    for variant in ("base", "sft"):
        reports = {}
        # One warmup pair plus three measured pairs. Reverse every second pair.
        for iteration in range(4):
            order = ("baseline", "cache") if iteration % 2 == 0 else ("cache", "baseline")
            for mode in order:
                graph = make_prompt(variant, "english", 42, 1024, 50, mode, tag=f"repeat-{iteration}")
                file = args.output / f"{variant}-{iteration}-{mode}.json"
                if file.exists():
                    report = json.loads(file.read_text(encoding="utf-8"))
                    if report.get("source_sha256") != source or report.get("prompt") != graph:
                        raise ValueError(f"Stale repeat evidence: {file}")
                else:
                    report = run(args.url, graph, file, args.log)
                if mode == "cache" and (not report["cache"] or report["cache"][-1]["hits"] == 0):
                    raise ValueError("Repeat measurement did not actually reuse blocks")
                if report["sampler_seconds"] is None:
                    raise ValueError("No actual KSampler execution was observed")
                reports[iteration, mode] = report
        reference = Image.open(image_path(reports[1, "baseline"], ROOT / "artifacts/comfy-output"))
        cache_reference = Image.open(image_path(reports[1, "cache"], ROOT / "artifacts/comfy-output"))
        baseline_noise = []
        cache_noise = []
        pairs = []
        for iteration in (1, 2, 3):
            baseline, cached = reports[iteration, "baseline"], reports[iteration, "cache"]
            baseline_image = Image.open(image_path(baseline, ROOT / "artifacts/comfy-output"))
            cached_image = Image.open(image_path(cached, ROOT / "artifacts/comfy-output"))
            baseline_noise.append(metrics(reference, baseline_image))
            cache_noise.append(metrics(cache_reference, cached_image))
            pairs.append({"iteration": iteration, "baseline_prompt_id": baseline["prompt_id"],
                          "cache_prompt_id": cached["prompt_id"],
                          "baseline_ksampler_s": baseline["sampler_seconds"],
                          "cache_ksampler_s": cached["sampler_seconds"],
                          "ksampler_speedup": baseline["sampler_seconds"] / cached["sampler_seconds"],
                          "baseline_total_s": baseline["elapsed_seconds"],
                          "cache_total_s": cached["elapsed_seconds"],
                          "cache": cached["cache"][-1]})
        speeds = [r["ksampler_speedup"] for r in pairs]
        groups.append({"variant": variant, "case": "english", "seed": 42,
                       "warmup_pairs_excluded": 1, "measured_pairs": 3,
                       "median_ksampler_speedup": statistics.median(speeds),
                       "range_ksampler_speedup": [min(speeds), max(speeds)],
                       "baseline_repeat_noise": baseline_noise, "cache_repeat_noise": cache_noise,
                       "pairs": pairs})
        (args.output / "progress.json").write_text(json.dumps({"completed_variants": len(groups)}, indent=2), encoding="utf-8")
    summary = {"source_sha256": source, "status": "completed", "groups": groups,
               "scope": "One warmup pair and three same-seed measured pairs per variant; 1024x1024, 50 steps, native PyTorch, alternating order. KSampler node wall time includes setup/loading."}
    (ROOT / "artifacts/repeats-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "medians": {g["variant"]: g["median_ksampler_speedup"] for g in groups}}))


if __name__ == "__main__":
    main()
