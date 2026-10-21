"""Execution-local, first-block guarded residual reuse for MOSS-Image."""

from collections import OrderedDict
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class CacheConfig:
    threshold: float = 0.08
    start_percent: float = 0.10
    end_percent: float = 0.90
    max_consecutive_hits: int = 1
    cache_device: str = "gpu"
    metric_stride: int = 4
    max_cache_mb: int = 512
    threshold_mode: str = "constant"
    split_ratio: float = 0.30
    late_threshold: float = 0.14

    def __post_init__(self):
        numbers = (self.threshold, self.start_percent, self.end_percent, self.split_ratio, self.late_threshold)
        if not all(math.isfinite(v) for v in numbers):
            raise ValueError("Cache parameters must be finite")
        if not 0 <= self.start_percent < self.end_percent <= 1:
            raise ValueError("Require 0 <= start_percent < end_percent <= 1")
        if not 0 <= self.threshold <= 1 or not 0 <= self.late_threshold <= 1:
            raise ValueError("Residual thresholds must be within [0, 1]")
        if self.threshold_mode not in ("constant", "two_stage") or not 0 <= self.split_ratio <= 1:
            raise ValueError("Invalid threshold mode or split ratio")
        if self.cache_device not in ("cpu", "gpu"):
            raise ValueError("cache_device must be cpu or gpu")
        if not 1 <= self.max_consecutive_hits <= 10 or not 1 <= self.metric_stride <= 32:
            raise ValueError("Invalid consecutive hit limit or metric stride")
        if not 1 <= self.max_cache_mb <= 32768:
            raise ValueError("max_cache_mb must be within [1, 32768]")


@dataclass
class Stream:
    sigma: float | None = None
    indicator: tuple | None = None
    tail: torch.Tensor | None = None
    consecutive: int = 0

    def clear(self):
        self.indicator = None
        self.tail = None
        self.consecutive = 0

    def bytes(self):
        tensors = list(self.indicator or ()) + ([self.tail] if self.tail is not None else [])
        return sum(t.numel() * t.element_size() for t in tensors)


class CacheRuntime:
    def __init__(self, config, model_sampling):
        self.config = config
        self.start = float(model_sampling.percent_to_sigma(config.start_percent))
        self.end = float(model_sampling.percent_to_sigma(config.end_percent))
        split = config.start_percent + (config.end_percent - config.start_percent) * config.split_ratio
        self.split = float(model_sampling.percent_to_sigma(split))
        self.budget = config.max_cache_mb * 1024 * 1024
        self.streams = OrderedDict()
        self.full = 0
        self.hits = 0
        self.skipped_blocks = 0
        self.peak_bytes = 0
        self.fallbacks = {}

    def fallback(self, reason):
        self.fallbacks[reason] = self.fallbacks.get(reason, 0) + 1

    def threshold(self, sigma):
        c = self.config
        return c.late_threshold if c.threshold_mode == "two_stage" and sigma <= self.split else c.threshold

    def active(self, sigma):
        return self.end < sigma <= self.start and self.threshold(sigma) > 0

    def stream(self, key, sigma):
        stream = self.streams.pop(key, Stream())
        if stream.sigma is not None and sigma >= stream.sigma:
            stream.clear()
        stream.sigma = sigma
        self.streams[key] = stream
        if len(self.streams) > 8:
            self.streams.popitem(last=False)
            self.fallback("cache_budget")
        return stream

    def copy(self, value):
        device = "cpu" if self.config.cache_device == "cpu" else value.device
        return value.detach().to(device=device, copy=True).contiguous()

    def sample(self, value, image_tokens):
        # Guard image and text independently; a long text must not dilute image changes.
        stride = self.config.metric_stride
        parts = (value[:, :image_tokens], value[:, image_tokens:])
        return tuple(self.copy(p[:, ::stride, ::stride].float()) for p in parts if p.shape[1])

    @staticmethod
    def difference(current, previous):
        if len(current) != len(previous):
            return math.inf
        scores = []
        for a, b in zip(current, previous):
            if a.shape != b.shape:
                return math.inf
            delta = (a - b).abs().mean((1, 2))
            scale = b.abs().mean((1, 2))
            score = delta / scale.clamp_min(1e-6)
            score = torch.where(torch.isfinite(delta) & torch.isfinite(scale), score, math.inf)
            worst = float(score.max())
            if not math.isfinite(worst):
                return math.inf
            scores.append(worst)
        return max(scores, default=math.inf)

    def replay(self, stream, indicator, sigma, anchor):
        if stream.tail is None or stream.indicator is None or not self.active(sigma):
            return None
        if stream.consecutive >= self.config.max_consecutive_hits:
            return None
        score = self.difference(indicator, stream.indicator)
        if not math.isfinite(score) or score >= self.threshold(sigma):
            return None
        result = anchor + stream.tail.to(anchor)
        if not bool(torch.isfinite(result).all()):
            stream.clear()
            self.fallback("nonfinite_replay")
            return None
        stream.consecutive += 1
        self.hits += 1
        return result

    def store(self, stream, indicator, output, anchor):
        tail = self.copy(output - anchor)
        if not bool(torch.isfinite(tail).all()) or any(not bool(torch.isfinite(v).all()) for v in indicator):
            stream.clear()
            self.fallback("nonfinite_history")
            return
        stream.tail = tail
        stream.indicator = indicator
        stream.consecutive = 0
        while self.streams and self.bytes() > self.budget:
            self.streams.popitem(last=False)
            self.fallback("cache_budget")
        self.peak_bytes = max(self.peak_bytes, self.bytes())

    def bytes(self):
        return sum(s.bytes() for s in self.streams.values())

    def clear(self):
        self.streams.clear()


@dataclass
class ForwardState:
    runtime: CacheRuntime
    stream: Stream
    sigma: float
    image_tokens: int
    block_count: int
    indicator: tuple | None = None
    anchor: torch.Tensor | None = None
    replay: torch.Tensor | None = None

