"""Serial, local ComfyUI verification. Uses existing models; never downloads weights."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
import uuid

import websocket


PROMPTS = {
    "photo": "A natural editorial photograph of a young adult woman in a blue wool coat, standing beside a red bicycle on a quiet autumn street, warm afternoon light, detailed face, hands and leaves, realistic photography.",
    "english": 'A clean studio photograph of a white enamel sign with large centered dark blue text exactly reading "MOSS IMAGE" on the first line and "OPEN 09:00" on the second line. A small green plant beside the sign, neutral wall, crisp typography.',
    "chinese": '一张现代咖啡馆门口的照片，白色招牌上清晰写着两行中文：第一行“春日咖啡”，第二行“欢迎光临”。招牌旁摆着一盆绿色植物，暖色阳光，文字端正清楚。',
    "anime": "A detailed anime illustration of an adult woman with short dark blue hair and amber eyes, wearing a white jacket, holding a small red umbrella on a rainy city street, vivid reflections, expressive face, clean hands, cinematic composition.",
    "people": "A realistic photograph of two adult friends seated across a small wooden cafe table, the woman on the left wearing green and the man on the right wearing blue, two white coffee cups between them, both faces visible, natural hands, soft window light.",
    "layout": json.dumps({"summary": {"caption": "A clean white travel poster with clear text and two colorful geometric objects."}, "items": [
        {"label": "text", "caption": "large black sans serif title", "text": "MOSS", "bbox": [-430, -350, -240, 350]},
        {"label": "object", "caption": "a red ceramic sphere", "bbox": [-100, -400, 300, -40]},
        {"label": "object", "caption": "a blue ceramic cube", "bbox": [-100, 40, 300, 400]},
    ], "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}),
}


def request(url, data=None):
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as e:
        raise RuntimeError(e.read().decode()) from e


def make_prompt(variant="sft", case="english", seed=42, size=1024, steps=50, mode="cache", threshold=0.08,
                skip22=True, cache_device="gpu", tag="verify", cfg=3.0):
    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": f"moss_image_original_{variant}_bf16.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "moss_qwen35_9b_bf16_native.safetensors", "type": "moss_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "moss_image_official_vae.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": PROMPTS[case]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": ""}},
        "6": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": size, "height": size, "batch_size": 1}},
        "10": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0], "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
        "12": {"class_type": "SaveImage", "inputs": {"images": ["11", 0], "filename_prefix": f"{tag}/{variant}-{case}-{seed}-{mode}"}},
    }
    # Native MOSS already has shift=5 and multiplier=1. ModelSamplingSD3 resets multiplier to 1000.
    model = ["1", 0]
    if mode != "baseline":
        graph["8"] = {"class_type": "MOSSImageBlockCacheT8", "inputs": {
            "model": model, "enabled": mode != "disabled", "threshold": 0.0 if mode == "zero" else threshold,
            "start_percent": 0.10, "end_percent": 0.90, "max_consecutive_hits": 1,
            "cache_device": cache_device, "max_cache_mb": 512, "metric_stride": 4,
            "threshold_mode": "constant", "split_ratio": 0.30, "late_threshold": 0.14,
        }}
        model = ["8", 0]
    if skip22:
        graph["9"] = {"class_type": "SkipLayerGuidanceDiTSimple", "inputs": {
            "model": model, "double_layers": "", "single_layers": "22", "start_percent": 0.0, "end_percent": 1.0,
        }}
        model = ["9", 0]
    graph["10"]["inputs"]["model"] = model
    return graph


def run(url, graph, destination, log_path=None):
    root = Path(__file__).resolve().parents[1]
    source = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for name in ("cache.py", "runtime.py", "nodes.py")}
    queue = request(url + "/queue")
    if queue.get("queue_running") or queue.get("queue_pending"):
        raise RuntimeError("Verification requires an idle dedicated ComfyUI queue")
    client = str(uuid.uuid4())
    ws = websocket.create_connection(url.replace("http", "ws", 1) + "/ws?clientId=" + client, timeout=30)
    offset = log_path.stat().st_size if log_path and log_path.exists() else 0
    start = time.perf_counter()
    prompt_id = request(url + "/prompt", {"prompt": graph, "client_id": client})["prompt_id"]
    print(json.dumps({"submitted": prompt_id, "report": str(destination)}), flush=True)
    events = []
    sampler_start = sampler_end = None
    try:
        while True:
            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                if prompt_id in request(url + "/history/" + prompt_id):
                    break
                continue
            if not isinstance(raw, str):
                continue
            message = json.loads(raw)
            data = message.get("data", {})
            if data.get("prompt_id") != prompt_id:
                continue
            kind = message["type"]
            elapsed = time.perf_counter() - start
            if kind == "executing":
                node = data.get("node")
                events.append({"node": node, "seconds": elapsed})
                if node == "10":
                    sampler_start = elapsed
                elif sampler_start is not None and sampler_end is None:
                    sampler_end = elapsed
                print(json.dumps({"node": node, "seconds": round(elapsed, 2)}), flush=True)
                if node is None:
                    break
            if kind == "execution_error":
                raise RuntimeError(json.dumps(data, ensure_ascii=False))
            if kind == "execution_interrupted":
                raise RuntimeError("Verification interrupted")
    finally:
        ws.close()
    history = request(url + "/history/" + prompt_id)[prompt_id]
    if history.get("status", {}).get("status_str") != "success":
        raise RuntimeError(json.dumps(history.get("status"), ensure_ascii=False))
    summaries = []
    if log_path:
        with log_path.open("rb") as f:
            f.seek(offset)
            logs = f.read().decode(errors="replace")
        summaries = [json.loads(v) for v in re.findall(r"MOSS BlockCache T8: (\{[^\r\n]+\})", logs)]
    if any(hashlib.sha256((root / name).read_bytes()).hexdigest() != digest for name, digest in source.items()):
        raise RuntimeError("Source changed during verification; this result must not be attributed to the new code")
    report = {"prompt_id": prompt_id, "elapsed_seconds": time.perf_counter() - start, "source_sha256": source,
              "sampler_seconds": None if sampler_start is None or sampler_end is None else sampler_end - sampler_start,
              "events": events, "cache": summaries, "outputs": history.get("outputs", {}), "prompt": graph,
              "system": request(url + "/system_stats")}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"complete": prompt_id, "sampler_seconds": report["sampler_seconds"], "cache": summaries}), flush=True)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8291")
    p.add_argument("--variant", choices=["base", "sft"], default="sft")
    p.add_argument("--case", choices=list(PROMPTS), default="english")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=3.0)
    p.add_argument("--mode", choices=["baseline", "cache", "zero", "disabled"], default="cache")
    p.add_argument("--threshold", type=float, default=0.08)
    p.add_argument("--cache-device", choices=["gpu", "cpu"], default="gpu")
    p.add_argument("--no-skip22", action="store_true")
    p.add_argument("--tag", default="verify")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--log", type=Path)
    args = p.parse_args()
    graph = make_prompt(args.variant, args.case, args.seed, args.size, args.steps, args.mode, args.threshold,
                        not args.no_skip22, args.cache_device, args.tag, args.cfg)
    run(args.url, graph, args.output, args.log)


if __name__ == "__main__":
    main()
