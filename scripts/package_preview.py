"""Build an honest, usable preview without relaxing the full-release gate."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET
import zipfile

from PIL import Image

from package import FILES, verify_workflows
from report_matrix import image_path, metrics


ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = {n: sha(ROOT/n) for n in ("cache.py", "runtime.py", "nodes.py")}
    verify_workflows()
    suites = list(ET.parse(ROOT / "artifacts/test-results.xml").getroot().iter("testsuite"))
    tests = sum(int(s.get("tests", 0)) for s in suites)
    failures = sum(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
    if tests < 51 or failures:
        raise ValueError("Preview requires passing native integration tests")
    canvas = load(ROOT / "artifacts/canvas-acceptance.json")
    for variant in ("base", "sft"):
        proof = canvas[variant]
        png = ROOT / "workflows" / f"MOSS_T8_{variant.upper()}_1024_BlockCache.png"
        if (proof.get("status") != "success" or proof.get("source_sha256") != source
            or proof.get("frontend_origin") != "comfyui-frontend" or proof.get("png_sha256") != sha(png)):
            raise ValueError("Both real canvas runs must cover the current preview source")
    gpu = load(ROOT / "artifacts/gpu-smoke.json")
    if gpu.get("status") != "passed" or gpu.get("source_sha256") != source:
        raise ValueError("Current-source GPU/CPU-cache smoke is required")
    zero = load(ROOT / "artifacts/final-zero-sft-english-42.json")
    native = load(ROOT / "artifacts/matrix/sft-english-42-baseline.json")
    if zero["source_sha256"] != source or native["source_sha256"] != source:
        raise ValueError("Stale zero-threshold evidence")
    parity = metrics(Image.open(image_path(zero, ROOT / "artifacts/comfy-output")),
                     Image.open(image_path(native, ROOT / "artifacts/comfy-output")))
    if not parity["pixel_equal"]:
        raise ValueError("Zero-threshold image differs from the native baseline")
    matrix = load(ROOT / "artifacts/matrix-summary.json")
    review = load(ROOT / "artifacts/manual-review.json")
    if matrix["source_sha256"] != source or review["source_sha256"] != source:
        raise ValueError("Stale partial quality evidence")
    rows = []
    for row in matrix["rows"]:
        key = f"{row['variant']}/{row['case']}/{row['seed']}"
        if key not in review["entries"]:
            continue
        note = review["entries"][key]
        if note["outcome"] != "acceptable_with_differences":
            raise ValueError(f"Resolve the rejected preview image: {key}")
        for mode in ("baseline", "cache"):
            report = load(ROOT / "artifacts/matrix" / f"{key.replace('/', '-')}-{mode}.json")
            if (report["source_sha256"] != source or report["prompt_id"] != note[f"{mode}_prompt_id"]
                or sha(image_path(report, ROOT / "artifacts/comfy-output")) != note[f"{mode}_png_sha256"]):
                raise ValueError("Review version does not match the actual image")
        rows.append({**row, "visual_review": note})
    if not rows:
        raise ValueError("Actual reviewed paired images are required")
    system = native["system"]["system"]
    summary = {"status": "preview", "version": "0.1.0-preview", "created_utc": datetime.now(timezone.utc).isoformat(),
               "source_sha256": source, "complete_36_pair_matrix": False,
               "tests": {"passed": tests, "failures_errors_skips": failures}, "canvas": canvas,
               "cuda_smoke": gpu, "zero_threshold_parity": parity,
               "manual_review": {"status": "partial", "reviewed_pairs": len(rows), "planned_pairs": 36,
                                 "reviewer": "Codex visual inspection of actual paired images"},
               "environment": {"comfyui_version": system["comfyui_version"],
                               "comfyui_commit": "8f04fd6ecf4a548937971e840af2a985529fab76",
                               "python": system["python_version"].split()[0], "pytorch": system["pytorch_version"],
                               "gpu": native["system"]["devices"][0]["name"], "attention": "pytorch"},
               "parameters": {"size": [1024, 1024], "steps": 50, "cfg": 3, "sampler": "euler",
                              "scheduler": "simple", "negative_skip22": True, "native_shift": 5,
                              "native_timestep_multiplier": 1, "threshold": 0.05, "window": [0.1, 0.9],
                              "max_consecutive_hits": 1, "cache_device": "gpu"},
               "pending": ["Complete remaining 36-pair quality matrix", "Same-seed warmup/repeated timings",
                           "Threshold/hit-cap/window sensitivity probes", "Full release acceptance"],
               "timing_scope": "KSampler node wall time includes setup/model loading; full workflow additionally includes encoding, decoding and saving. Partial reviewed SFT samples only; no general or Base performance claim.",
               "rows": rows}
    validation = ROOT / "validation/preview-summary.json"
    validation.parent.mkdir(exist_ok=True)
    validation.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    speeds = [r["sampling_speedup"] for r in rows]
    total = [r["baseline_total_s"] / r["cache_total_s"] for r in rows]
    coverage = ", ".join(sorted({f"{r['variant'].upper()}/{r['case']}" for r in rows}))
    lines = ["# 预览版验证记录 / Preview validation", "",
             "本记录属于 0.1.0-preview。完整 36 组成对画质、参数敏感性和重复测速验收仍在进行。", "",
             f"- 当前源码通过 {tests} 项自动化测试，0 failures/errors/skips。",
             "- Base、SFT 均在真实 ComfyUI 画布点击运行成功，保存 JSON 和内嵌完整工作流的 PNG，并重新导入验证。",
             "- 完整权重 SFT：零阈值与未接插件的原生 1024² / 50 步基线逐像素一致。",
             "- 真实微型 MOSS CUDA BF16：GPU/CPU 缓存输出一致，实际跳层和输出头调用验证通过；此项不代表完整权重 CPU 缓存性能。", "",
             f"已实际检查 {len(rows)} 组配对图片，覆盖 {coverage}，记录在 [preview-summary.json](validation/preview-summary.json)。当前已审样本的 KSampler 节点中位数加速 {statistics.median(speeds):.3f}×（范围 {min(speeds):.3f}–{max(speeds):.3f}×），整工作流中位数 {statistics.median(total):.3f}×。这是有限样本结果，不代表 Base 的测速结论。", "",
             "## 测试条件", "",
             f"ComfyUI {system['comfyui_version']} / `{summary['environment']['comfyui_commit']}`；Python {summary['environment']['python']}；PyTorch {system['pytorch_version']}；{summary['environment']['gpu']}；原生 PyTorch attention。",
             "本地已有原生 BF16 Base/SFT、Qwen3.5-9B TE 和官方 MOSS VAE。1024×1024，50 步，Euler/simple，CFG3，negative skip22，原生 shift5 / timestep multiplier1；threshold0.05，窗口0.10–0.90，最多连续命中1次。未下载或修改模型。",
             "专用测试服务关闭执行结果缓存、dynamic VRAM 和 Comfy compiler。KSampler 节点耗时包含 setup / 模型装载；整流程另含文本编码、VAE 解码和保存。", "",
             "## 已观察到的限制", "",
             "缓存是近似复用。已审照片中面部、发丝和细节会变化；中文招牌文字保留，但植物比例、背景灯具、装饰和反光可明显变化。部分英文原生基线已有换行偏差，缓存图保留该偏差。像素指标不能替代文字、人物和布局检查。",
             "本预览不提供其他 GPU、量化/FP16 权重、编译或 offload 模式的速度与质量结论。完整测试结束后再形成最终版本报告。", ""]
    (ROOT / "BENCHMARKS.md").write_text("\n".join(lines), encoding="utf-8")
    files = [ROOT/n for n in FILES] + sorted((ROOT/"workflows").glob("*.json")) + sorted((ROOT/"workflows").glob("*.png")) + [validation]
    target = ROOT / "dist/comfyui-moss-image-blockcache-T8-0.1.0-preview.zip"
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            if not file.is_file():
                raise ValueError(f"Missing preview file: {file.name}")
            archive.write(file, "comfyui-moss-image-blockcache-T8/" + file.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None or any("roadmap" in n.lower() or "/models/" in n or "/artifacts/" in n or "/.env" in n for n in archive.namelist()):
            raise ValueError("Invalid preview archive")
    target.with_suffix(".zip.sha256").write_text(sha(target) + "  " + target.name + "\n", encoding="ascii")
    print(json.dumps({"archive": str(target), "sha256": sha(target), "reviewed_pairs": len(rows), "status": "preview"}))


if __name__ == "__main__":
    main()
