"""Finalize existing evidence within the user's finite scope; no GPU jobs."""

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET

from PIL import Image

from package import verify_summary, verify_workflows
from report_matrix import image_path, metrics


ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = {n: digest(ROOT/n) for n in ("cache.py", "runtime.py", "nodes.py")}
    matrix = load(ROOT / "artifacts/matrix-summary.json")
    reviews = load(ROOT / "artifacts/manual-review.json")
    if matrix["source_sha256"] != source or reviews["source_sha256"] != source:
        raise ValueError("Reviewed evidence must cover the current source")
    rows = []
    for row in matrix["rows"]:
        key = f"{row['variant']}/{row['case']}/{row['seed']}"
        review = reviews["entries"].get(key)
        if review is None:
            continue
        if review.get("outcome") != "acceptable_with_differences" or not review.get("notes"):
            raise ValueError(f"Unresolved image review: {key}")
        for mode in ("baseline", "cache"):
            report = load(ROOT / "artifacts/matrix" / f"{key.replace('/', '-')}-{mode}.json")
            if (report["source_sha256"] != source or report["prompt_id"] != review[f"{mode}_prompt_id"]
                or digest(image_path(report, ROOT / "artifacts/comfy-output")) != review[f"{mode}_png_sha256"]):
                raise ValueError(f"Reviewed image version does not match: {key}/{mode}")
        rows.append({**row, "visual_review": review})
    suites = list(ET.parse(ROOT / "artifacts/test-results.xml").getroot().iter("testsuite"))
    tests = sum(int(s.get("tests", 0)) for s in suites)
    failures = sum(int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped"))
    zero = load(ROOT / "artifacts/final-zero-sft-english-42.json")
    baseline = load(ROOT / "artifacts/matrix/sft-english-42-baseline.json")
    if zero["source_sha256"] != source or baseline["source_sha256"] != source:
        raise ValueError("Zero-threshold evidence must cover the current source")
    with Image.open(image_path(zero, ROOT / "artifacts/comfy-output")) as a, Image.open(
        image_path(baseline, ROOT / "artifacts/comfy-output")) as b:
        parity = metrics(a, b)
    system = baseline["system"]["system"]
    summary = {
        "status": "completed", "version": "0.1.0", "created_utc": datetime.now(timezone.utc).isoformat(),
        "validation_scope": "user_approved_finite", "complete_36_pair_matrix": False,
        "acceptance_basis": "User requested one or two successful test groups and stopping extended testing. Both Base/SFT real canvas runs and existing reviewed evidence were retained.",
        "extended_testing": {"status": "stopped_per_user_request",
                             "not_completed": ["Full 36-pair matrix", "Same-seed repeat timings", "Parameter sensitivity probes"]},
        "source_sha256": source, "tests": {"passed": tests, "failures_errors_skips": failures},
        "canvas": load(ROOT / "artifacts/canvas-acceptance.json"),
        "cuda_smoke": load(ROOT / "artifacts/gpu-smoke.json"), "zero_threshold_parity": parity,
        "manual_review": {"status": "passed", "reviewed_pairs": len(rows),
                          "scope": "Only the actual reviewed rows below; no full-matrix acceptance claim",
                          "reviewer": "Codex visual inspection of actual paired images"},
        "environment": {"comfyui_version": system["comfyui_version"],
                        "comfyui_commit": "8f04fd6ecf4a548937971e840af2a985529fab76",
                        "python": system["python_version"].split()[0], "pytorch": system["pytorch_version"],
                        "gpu": baseline["system"]["devices"][0]["name"], "attention": "pytorch"},
        "parameters": {"size": [1024, 1024], "steps": 50, "cfg": 3, "sampler": "euler",
                       "scheduler": "simple", "negative_skip_layer": 22, "native_shift": 5,
                       "native_timestep_multiplier": 1, "threshold": 0.05, "window": [0.1, 0.9],
                       "max_consecutive_hits": 1, "cache_device": "gpu"},
        "timing_scope": "KSampler-node wall time includes setup/model loading; workflow time also includes encoding, decoding and saving. Baseline precedes cache. Finite reviewed SFT samples, not pure GPU kernel time, randomized repeated timing, or a Base speed claim.",
        "rows": rows,
    }
    verify_summary(summary)
    verify_workflows()
    target = ROOT / "validation/summary.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    speed = [r["sampling_speedup"] for r in rows]
    total = [r["baseline_total_s"] / r["cache_total_s"] for r in rows]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["variant"], row["case"])].append(row)
    lines = ["# 实测验证 / Measured validation — v0.1.0", "",
             "按用户要求采用有限验证，并已停止扩展跑图。完整 36 组矩阵、同 seed 重复计时和参数敏感性探测未完成，不作为本次交付的完成条件。", "",
             f"- {tests} 项自动化测试通过，0 failures/errors/skips；包含真实微型原生 MOSS 的跳层、输出头、回退、清理和工作流检查。",
             "- Base、SFT 均在真实 ComfyUI 画布点击运行成功，保存 JSON 和内嵌工作流的 PNG，并重新导入验证。",
             "- 完整权重 SFT 1024² / 50 步零阈值结果与未接插件的原生基线逐像素一致。",
             "- 真实微型原生 MOSS CUDA BF16：GPU/CPU 缓存输出一致；不代表完整权重 CPU 缓存性能。", "",
             f"已经实际检查 {len(rows)} 组成对样本。KSampler 节点中位数加速 **{statistics.median(speed):.3f}×**，范围 {min(speed):.3f}–{max(speed):.3f}×；整工作流中位数 {statistics.median(total):.3f}×。这是有限 SFT 样本的观测，不能推广为 Base 或其他设备的速度保证。", "",
             "## 测试条件", "",
             f"ComfyUI {system['comfyui_version']} / `{summary['environment']['comfyui_commit']}`；Python {summary['environment']['python']}；PyTorch {system['pytorch_version']}；{summary['environment']['gpu']}；原生 PyTorch attention。",
             "使用本地已有原生 BF16 Base/SFT、Qwen3.5-9B 文本编码器和官方 MOSS VAE，无模型下载或转换。1024×1024，50 步，Euler/simple，CFG3，negative skip22；原生 shift5 / multiplier1，不接 ModelSamplingSD3。缓存阈值0.05，窗口0.10–0.90，最多连续命中1次。",
             "测试服务关闭执行结果缓存、dynamic VRAM 和 Comfy compiler。KSampler 节点计时包含 setup 和模型加载；整流程另含编码、解码和保存。基线先于缓存运行，未做随机顺序的同 seed 重复测速。", "",
             "| 模型 | 场景 | 已审对数 | KSampler 中位数加速 | 整流程中位数加速 |", "|---|---|---:|---:|---:|"]
    for (variant, case), group in sorted(groups.items()):
        lines.append(f"| {variant.upper()} | {case} | {len(group)} | {statistics.median(r['sampling_speedup'] for r in group):.3f}× | {statistics.median(r['baseline_total_s']/r['cache_total_s'] for r in group):.3f}× |")
    lines += ["", "## 已观察到的限制", "",
              "近似缓存会改变面部、头发、服装配件、手部姿态、杯子比例、背景灯具和植物比例。已审中文招牌主要文字保留；部分英文原生基线已有换行偏差，缓存保留该偏差。布局样本保留主体位置，轮廓和高光存在细微变化。逐组记录、prompt ID 和图片哈希见 [validation/summary.json](validation/summary.json)。像素指标不能证明文字准确或美学质量。",
              "本次验收范围为原生 dense MOSS Base/SFT BF16、普通文生图及 JSON Layout、eager PyTorch 后端和本机 GPU。没有其他 GPU、量化/FP16、编译、低显存 offload、参考图编辑或 HF pipeline 一致性的实测结论。", ""]
    (ROOT / "BENCHMARKS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "completed", "reviewed_pairs": len(rows), "tests": tests,
                      "median_ksampler_speedup": statistics.median(speed)}))


if __name__ == "__main__":
    main()
