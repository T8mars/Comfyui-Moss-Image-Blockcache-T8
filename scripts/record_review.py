"""Bind an explicitly supplied visual-review decision to actual output versions."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from report_matrix import image_path


ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=["base", "sft"])
    parser.add_argument("case", choices=["photo", "english", "chinese", "anime", "people", "layout"])
    parser.add_argument("seed", type=int, choices=[42, 43, 44])
    parser.add_argument("--outcome", required=True, choices=["acceptable_with_differences", "rejected"])
    parser.add_argument("--notes", required=True)
    parser.add_argument("--viewed", required=True)
    args = parser.parse_args()
    if not args.notes.strip() or not args.viewed.strip():
        raise ValueError("Describe the actual inspection and differences")
    source = {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ("cache.py", "runtime.py", "nodes.py")}
    path = ROOT / "artifacts/manual-review.json"
    ledger = load(path) if path.exists() else {"source_sha256": source, "entries": {}}
    if ledger["source_sha256"] != source:
        raise ValueError("Archive the previous-source review ledger first")
    key = f"{args.variant}/{args.case}/{args.seed}"
    entry = {"outcome": args.outcome, "notes": args.notes, "viewed": args.viewed,
             "reviewed_utc": datetime.now(timezone.utc).isoformat()}
    for mode in ("baseline", "cache"):
        report = load(ROOT / "artifacts/matrix" / f"{key.replace('/', '-')}-{mode}.json")
        if report["source_sha256"] != source:
            raise ValueError("Review must cover current-source paired images")
        output = image_path(report, ROOT / "artifacts/comfy-output")
        entry[f"{mode}_prompt_id"] = report["prompt_id"]
        entry[f"{mode}_png_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    ledger["entries"][key] = entry
    path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    state_path = ROOT / "artifacts/development-state.json"
    if state_path.exists():
        state = load(state_path)
        state["manual_quality_review"]["reviewed_pairs"] = len(ledger["entries"])
        progress_path = ROOT / "artifacts/matrix/progress.json"
        if progress_path.exists():
            state["last_verified_matrix_progress"] = load(progress_path)
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"recorded": key, "outcome": args.outcome, "reviewed_pairs": len(ledger["entries"])}))


if __name__ == "__main__":
    main()
