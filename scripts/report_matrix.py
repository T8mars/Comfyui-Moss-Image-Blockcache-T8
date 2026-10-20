"""Compare paired pixels and prepare contact sheets for manual quality review."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
from PIL import Image, ImageDraw


def image_path(report, output_root):
    images = [i for node in report["outputs"].values() for i in node.get("images", [])]
    if len(images) != 1:
        raise ValueError("Expected exactly one saved output")
    item = images[0]
    return output_root / item["subfolder"] / item["filename"]


def metrics(a, b):
    x = np.asarray(a.convert("RGB"), dtype=np.float64)
    y = np.asarray(b.convert("RGB"), dtype=np.float64)
    if x.shape != y.shape or x.shape != (1024, 1024, 3):
        raise ValueError("Unexpected image dimensions")
    mse = float(np.square(x-y).mean())
    return {"mae_255": float(np.abs(x-y).mean()), "rmse_255": math.sqrt(mse),
            "psnr_db": None if mse == 0 else 10*math.log10(255**2/mse),
            "pixel_equal": bool(np.array_equal(x,y)), "baseline_std": float(x.std()), "cache_std": float(y.std())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("artifacts/matrix"))
    p.add_argument("--images", type=Path, default=Path("artifacts/comfy-output"))
    p.add_argument("--output", type=Path, default=Path("artifacts/matrix-summary.json"))
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    expected_hash = {n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ("cache.py","runtime.py","nodes.py")}
    rows = []
    groups = {}
    for file in sorted(args.input.glob("*-baseline.json")):
        cached_file = file.with_name(file.name.replace("-baseline.json", "-cache.json"))
        if not cached_file.exists():
            continue
        baseline = json.loads(file.read_text(encoding="utf-8"))
        cached = json.loads(cached_file.read_text(encoding="utf-8"))
        assert baseline["source_sha256"] == cached["source_sha256"] == expected_hash
        variant, case, seed, _ = file.stem.split("-")
        a = Image.open(image_path(baseline, args.images))
        b = Image.open(image_path(cached, args.images))
        score = metrics(a,b)
        assert score["baseline_std"] > 1 and score["cache_std"] > 1
        cache = cached["cache"][-1]
        assert cache["hits"] > 0 and cache["skipped_blocks"] > 0
        row = {"variant": variant, "case": case, "seed": int(seed), **score,
               "baseline_sampler_s": baseline["sampler_seconds"], "cache_sampler_s": cached["sampler_seconds"],
               "baseline_total_s": baseline["elapsed_seconds"], "cache_total_s": cached["elapsed_seconds"],
               "sampling_speedup": baseline["sampler_seconds"] / cached["sampler_seconds"],
               "cache": cache, "baseline_prompt_id": baseline["prompt_id"], "cache_prompt_id": cached["prompt_id"]}
        rows.append(row)
        groups.setdefault((variant,case), []).append((row,a.copy(),b.copy()))
    contacts = args.input / "contact-sheets"
    contacts.mkdir(exist_ok=True)
    for (variant,case), entries in groups.items():
        sheet = Image.new("RGB", (800, len(entries)*428), "#f0f0f0")
        draw = ImageDraw.Draw(sheet)
        for index,(row,a,b) in enumerate(entries):
            y = index*428
            for x,im,label in [(0,a,"native baseline"),(400,b,"BlockCache 0.05")]:
                draw.text((x+8,y+5), f"{variant}/{case} seed={row['seed']} {label}",fill="black")
                thumb=im.resize((384,384))
                sheet.paste(thumb,(x+8,y+26))
            draw.text((8,y+412),f"MAE={row['mae_255']:.3f}/255; speedup={row['sampling_speedup']:.3f}; hits={row['cache']['hits']}",fill="black")
        sheet.save(contacts/f"{variant}-{case}.png")
    summary = {"pairs": len(rows), "complete_36_pair_matrix": len(rows)==36,
               "source_sha256": expected_hash, "rows": rows,
               "median_sampling_speedup": statistics.median(r["sampling_speedup"] for r in rows) if rows else None,
               "quality_review": "Pixel metrics are supplemental; text, layout and people require manual review of the actual images."}
    args.output.write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k not in ("rows","source_sha256")},indent=2))


if __name__ == "__main__":
    main()
