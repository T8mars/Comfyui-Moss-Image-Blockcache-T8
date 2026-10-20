import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("t8_release", ROOT / "scripts" / "package.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    for name in release.FILES:
        source = ROOT / name
        target = tmp_path / name
        if source.exists():
            shutil.copy2(source, target)
        else:
            target.write_text("Package-verifier fixture only", encoding="utf-8")
    shutil.copytree(ROOT / "workflows", tmp_path / "workflows")
    (tmp_path / "validation").mkdir()
    summary = {"complete_36_pair_matrix": True,
               "rows": [{"variant": v, "case": c, "seed": s}
                        for v in ("base", "sft") for c in ("photo", "english", "chinese", "anime", "people", "layout")
                        for s in (42, 43, 44)],
               "manual_review": {"status": "passed", "reviewed_pairs": 36},
               "source_sha256": {n: hashlib.sha256((tmp_path/n).read_bytes()).hexdigest()
                                 for n in ("cache.py", "runtime.py", "nodes.py")}}
    (tmp_path / "validation" / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    monkeypatch.setattr(release, "ROOT", tmp_path)
    return tmp_path, summary


def test_package_excludes_local_plan_models_and_secrets(prepared):
    root, _ = prepared
    (root / "roadmap.md").write_text("PRIVATE RESEARCH", encoding="utf-8")
    (root / ".env").write_text("DUMMY_TEST_SECRET=not-real", encoding="utf-8")
    (root / "models").mkdir()
    (root / "models" / "dummy.safetensors").write_bytes(b"not a model")
    release.main()
    target = next((root / "dist").glob("*.zip"))
    with zipfile.ZipFile(target) as archive:
        names = archive.namelist()
        assert any(n.endswith("/runtime.py") for n in names)
        assert sum(n.endswith(".png") for n in names) == 2
        assert not any("roadmap" in n or ".env" in n or "models/" in n for n in names)
        assert archive.testzip() is None


def test_package_rejects_incomplete_matrix(prepared):
    root, summary = prepared
    summary["rows"].pop()
    (root / "validation" / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="matrix is incomplete"):
        release.main()


def test_package_rejects_unreviewed_images(prepared):
    _, summary = prepared
    summary["manual_review"] = {"status": "pending", "reviewed_pairs": 0}
    with pytest.raises(ValueError, match="Manual review"):
        release.verify_summary(summary)


def test_package_rejects_stale_code_evidence(prepared):
    root, summary = prepared
    (root / "runtime.py").write_text("Changed plugin code", encoding="utf-8")
    with pytest.raises(ValueError, match="current plugin source"):
        release.verify_summary(summary)


def test_package_rejects_wrong_canvas_configuration(prepared):
    root, _ = prepared
    file = root / "workflows" / "MOSS_T8_SFT_1024_BlockCache.json"
    graph = json.loads(file.read_text(encoding="utf-8"))
    node = next(n for n in graph["nodes"] if n["type"] == "MOSSImageBlockCacheT8")
    node["widgets_values"][1] = 0.9
    file.write_text(json.dumps(graph), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON and generated PNG disagree"):
        release.verify_workflows()


@pytest.fixture
def finite_summary(prepared):
    root, summary = prepared
    summary.update(validation_scope="user_approved_finite", complete_36_pair_matrix=False)
    summary["rows"] = summary["rows"][:2]
    for row in summary["rows"]:
        row["visual_review"] = {"outcome": "acceptable_with_differences", "notes": "Test fixture only"}
    summary["manual_review"]["reviewed_pairs"] = 2
    summary["tests"] = {"passed": 51, "failures_errors_skips": 0}
    summary["zero_threshold_parity"] = {"pixel_equal": True}
    summary["cuda_smoke"] = {"status": "passed", "source_sha256": summary["source_sha256"]}
    summary["canvas"] = {
        variant: {"status": "success", "frontend_origin": "comfyui-frontend",
                  "source_sha256": summary["source_sha256"],
                  "png_sha256": hashlib.sha256((root / "workflows" / f"MOSS_T8_{variant.upper()}_1024_BlockCache.png").read_bytes()).hexdigest()}
        for variant in ("base", "sft")}
    return summary


def test_finite_acceptance_uses_real_canvas_and_actual_pair_count(finite_summary):
    summary = finite_summary
    release.verify_summary(summary)
    summary["canvas"]["base"]["png_sha256"] = "stale-image"
    with pytest.raises(ValueError, match="real canvas evidence"):
        release.verify_summary(summary)


def test_finite_acceptance_rejects_false_full_matrix_claim(finite_summary):
    summary = finite_summary
    summary["complete_36_pair_matrix"] = True
    with pytest.raises(ValueError, match="actual unique reviewed pairs"):
        release.verify_summary(summary)


def test_package_filename_tracks_project_version(prepared):
    root, _ = prepared
    metadata = root / "pyproject.toml"
    metadata.write_text('[project]\nversion = "9.8.7"\n', encoding="utf-8")
    release.main()
    assert (root / "dist/comfyui-moss-image-blockcache-T8-9.8.7.zip").is_file()
