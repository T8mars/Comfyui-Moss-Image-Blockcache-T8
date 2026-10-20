"""Small, paired sensitivity probes for threshold, hit cap and active window."""

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

from report_matrix import image_path, metrics
from verify_native import make_prompt, run


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = {
    "lower_threshold": {"threshold": 0.01},
    "two_hits": {"max_consecutive_hits": 2},
    "narrow_window": {"start_percent": 0.2, "end_percent": 0.8},
}


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    source = {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ("cache.py", "runtime.py", "nodes.py")}
    progress = load(ROOT / "artifacts/matrix/progress.json")
    deployment = load(ROOT / "artifacts/server-deployment.json")
    if progress.get("completed") != 72 or progress["source_sha256"] != source or deployment["source_sha256"] != source:
        raise ValueError("Finish the final-source matrix on the dedicated server first")
    destination = ROOT / "artifacts/calibration"
    destination.mkdir(exist_ok=True)
    rows = []
    for case in ("chinese", "photo"):
        baseline = load(ROOT / f"artifacts/matrix/sft-{case}-42-baseline.json")
        default = load(ROOT / f"artifacts/matrix/sft-{case}-42-cache.json")
        if baseline["source_sha256"] != source or default["source_sha256"] != source:
            raise ValueError("Calibration baseline does not cover the current source")
        a = Image.open(image_path(baseline, ROOT / "artifacts/comfy-output"))
        images = [("native", a.copy()), ("default 0.05 / 1 hit / 0.1-0.9", Image.open(image_path(default, ROOT / "artifacts/comfy-output")))]
        for name, overrides in CANDIDATES.items():
            graph = make_prompt("sft", case, 42, 1024, 50, "cache", tag=f"calibration-{name}")
            graph["8"]["inputs"].update(overrides)
            report_path = destination / f"sft-{case}-42-{name}.json"
            if report_path.exists():
                report = load(report_path)
                if report.get("source_sha256") != source or report.get("prompt") != graph:
                    raise ValueError(f"Stale calibration result: {report_path}")
            else:
                report = run("http://127.0.0.1:8291", graph, report_path, ROOT / "artifacts/server-8291-error.log")
            b = Image.open(image_path(report, ROOT / "artifacts/comfy-output"))
            rows.append({"variant": "sft", "case": case, "seed": 42, "candidate": name, "overrides": overrides,
                         "cache": report["cache"][-1], "ksampler_node_speedup": baseline["sampler_seconds"] / report["sampler_seconds"],
                         "ksampler_node_seconds": report["sampler_seconds"],
                         "baseline_prompt_id": baseline["prompt_id"], "candidate_prompt_id": report["prompt_id"],
                         "candidate_png_sha256": hashlib.sha256(image_path(report, ROOT / "artifacts/comfy-output").read_bytes()).hexdigest(),
                         **metrics(a, b)})
            images.append((name, b.copy()))
        sheet = Image.new("RGB", (5 * 400, 428), "#f0f0f0")
        draw = ImageDraw.Draw(sheet)
        for index, (label, image) in enumerate(images):
            draw.text((index * 400 + 8, 5), label, fill="black")
            sheet.paste(image.resize((384, 384)), (index * 400 + 8, 26))
        sheet.save(destination / f"sft-{case}-42-comparison.png")
    summary = {"status": "completed", "source_sha256": source, "rows": rows,
               "scope": "Sensitivity probes on SFT seed42 photo/Chinese only. Each changes one control relative to defaults. Not a broad preset or optimal-threshold claim."}
    (ROOT / "artifacts/calibration-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "probes": len(rows)}))


if __name__ == "__main__":
    main()
