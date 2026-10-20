"""Build a small installable archive from an explicit public-file allowlist."""
from pathlib import Path
import hashlib
import json
import re
import zipfile

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FILES = ["__init__.py", "nodes.py", "cache.py", "runtime.py", "pyproject.toml", "README.md", "README_EN.md",
         "LICENSE", "NOTICE.md", "BENCHMARKS.md", ".comfyignore"]


def canonical(workflow):
    nodes = sorted((n["id"], n["type"], n.get("mode", 0), n.get("widgets_values", [])) for n in workflow["nodes"])
    return nodes, sorted(workflow["links"])


def verify_workflows():
    for variant in ("BASE", "SFT"):
        stem = ROOT / "workflows" / f"MOSS_T8_{variant}_1024_BlockCache"
        workflow = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
        with Image.open(stem.with_suffix(".png")) as image:
            embedded = json.loads(image.info["workflow"])
            if image.size != (1024, 1024) or canonical(workflow) != canonical(embedded):
                raise ValueError(f"Canvas JSON and generated PNG disagree: {variant}")
        if len(workflow["nodes"]) != 12 or len(workflow["links"]) != 11:
            raise ValueError(f"Unexpected workflow connections: {variant}")
        if sum(n["type"] == "MOSSImageBlockCacheT8" for n in workflow["nodes"]) != 1:
            raise ValueError(f"Missing or duplicate cache node: {variant}")
        if any(n["type"].startswith("ModelSampling") for n in workflow["nodes"]):
            raise ValueError(f"MOSS native schedule was replaced: {variant}")
        api = json.loads(Path(str(stem) + "-api.json").read_text(encoding="utf-8"))
        loaders = [n for n in api.values() if n["class_type"] == "UNETLoader"]
        if len(loaders) != 1 or variant.lower() not in loaders[0]["inputs"]["unet_name"].lower():
            raise ValueError(f"Wrong variant in API workflow: {variant}")
        for node in api.values():
            for value in node["inputs"].values():
                if isinstance(value, list) and str(value[0]) not in api:
                    raise ValueError(f"Dangling API workflow connection: {variant}")


def verify_summary(summary):
    expected = {(v,c,s) for v in ("base","sft") for c in ("photo","english","chinese","anime","people","layout")
                for s in (42,43,44)}
    rows = summary.get("rows", [])
    actual = {(r["variant"],r["case"],r["seed"]) for r in rows}
    review = summary.get("manual_review", {})
    current = {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ("cache.py","runtime.py","nodes.py")}
    if summary.get("source_sha256") != current:
        raise ValueError("Verification does not cover the current plugin source")
    if summary.get("validation_scope") == "user_approved_finite":
        if summary.get("complete_36_pair_matrix") or len(rows) < 2 or len(actual) != len(rows):
            raise ValueError("Finite validation must report actual unique reviewed pairs")
        if review.get("status") != "passed" or review.get("reviewed_pairs") != len(rows):
            raise ValueError("Manual review of actual paired images is incomplete")
        if any(not r.get("visual_review", {}).get("notes") or
               r["visual_review"].get("outcome") != "acceptable_with_differences" for r in rows):
            raise ValueError("Manual review notes are required for each reported pair")
        tests = summary.get("tests", {})
        if tests.get("passed", 0) < 51 or tests.get("failures_errors_skips") != 0:
            raise ValueError("Passing native integration tests are required")
        if not summary.get("zero_threshold_parity", {}).get("pixel_equal"):
            raise ValueError("Native zero-threshold parity is required")
        smoke = summary.get("cuda_smoke", {})
        if smoke.get("status") != "passed" or smoke.get("source_sha256") != current:
            raise ValueError("Current-source CUDA smoke is required")
        for variant in ("base", "sft"):
            proof = summary.get("canvas", {}).get(variant, {})
            png = ROOT / "workflows" / f"MOSS_T8_{variant.upper()}_1024_BlockCache.png"
            if (proof.get("status") != "success" or proof.get("frontend_origin") != "comfyui-frontend"
                or proof.get("source_sha256") != current
                or proof.get("png_sha256") != hashlib.sha256(png.read_bytes()).hexdigest()):
                raise ValueError(f"Current-source real canvas evidence is required: {variant}")
    else:
        if not summary.get("complete_36_pair_matrix") or len(rows) != 36 or actual != expected:
            raise ValueError("The paired verification matrix is incomplete")
        if review.get("status") != "passed" or review.get("reviewed_pairs") != 36:
            raise ValueError("Manual review of actual paired images is incomplete")


def main():
    files = [ROOT / name for name in FILES]
    files += sorted((ROOT / "workflows").glob("*.json"))
    files += sorted((ROOT / "workflows").glob("*.png"))
    summary = ROOT / "validation" / "summary.json"
    if not summary.is_file():
        raise ValueError("A completed validation/summary.json is required before packaging")
    verify_summary(json.loads(summary.read_text(encoding="utf-8")))
    verify_workflows()
    files.append(summary)
    files += sorted(p for p in (ROOT / "validation").glob("review-*.json")
                    if re.fullmatch(r"review-\d+\.\d+\.\d+\.json", p.name))
    missing = [str(p.relative_to(ROOT)) for p in files if not p.is_file()]
    if missing:
        raise ValueError(f"Missing release files: {missing}")
    for variant in ("BASE", "SFT"):
        for suffix in (".json", ".png", "-api.json"):
            required = ROOT / "workflows" / f"MOSS_T8_{variant}_1024_BlockCache{suffix}"
            if required not in files:
                raise ValueError(f"Missing verified workflow: {required.name}")
    folder = "comfyui-moss-image-blockcache-T8"
    version = re.search(r'^version\s*=\s*"(\d+\.\d+\.\d+)"\s*$',
                        (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.MULTILINE)
    if version is None:
        raise ValueError("A three-part project version is required")
    target = ROOT / "dist" / f"{folder}-{version.group(1)}.zip"
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            archive.write(file, f"{folder}/{file.relative_to(ROOT).as_posix()}")
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        assert not any("roadmap" in n.lower() or "/models/" in n or "/artifacts/" in n for n in archive.namelist())
    target.with_suffix(".zip.sha256").write_text(hashlib.sha256(target.read_bytes()).hexdigest() + "  " + target.name + "\n", encoding="ascii")
    print(target)


if __name__ == "__main__":
    main()
