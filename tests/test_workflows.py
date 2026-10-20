import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("verify_native", ROOT / "scripts" / "verify_native.py")
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


def test_native_schedule_is_preserved():
    for mode in ("baseline", "zero", "disabled", "cache"):
        graph = verify.make_prompt(mode=mode)
        assert not any(n["class_type"].startswith("ModelSampling") for n in graph.values())
        assert graph["9"]["inputs"]["single_layers"] == "22"
        assert graph["6"]["class_type"] == "EmptyFlux2LatentImage"
        assert graph["2"]["inputs"]["type"] == "moss_image"


def test_frontend_exports_are_connected_and_contain_no_private_plan():
    files = list((ROOT / "workflows").glob("*-api.json"))
    assert len(files) == 2
    for file in files:
        graph = json.loads(file.read_text(encoding="utf-8"))
        for node in graph.values():
            for value in node["inputs"].values():
                if isinstance(value, list):
                    assert str(value[0]) in graph
        assert not any(n["class_type"].startswith("ModelSampling") for n in graph.values())
        text = file.read_text(encoding="utf-8")
        assert "roadmap" not in text and "E:/" not in text and "E:\\\\" not in text
