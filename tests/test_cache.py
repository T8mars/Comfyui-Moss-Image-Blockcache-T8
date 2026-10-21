from dataclasses import replace
import math

import pytest
import torch

from moss_blockcache_t8.cache import CacheConfig, CacheRuntime


class Sampling:
    def percent_to_sigma(self, percent):
        t = 1 - percent
        return 5 * t / (1 + 4 * t)


def runtime(**kwargs):
    return CacheRuntime(replace(CacheConfig(), **kwargs), Sampling())


@pytest.mark.parametrize("values", [
    {"start_percent": 0.9, "end_percent": 0.1}, {"threshold": math.nan}, {"late_threshold": math.inf},
    {"threshold": -0.1}, {"max_consecutive_hits": 0}, {"metric_stride": 0}, {"cache_device": "disk"},
    {"max_cache_mb": 0}, {"threshold_mode": "unknown"}, {"split_ratio": 2},
])
def test_invalid_config(values):
    with pytest.raises(ValueError):
        replace(CacheConfig(), **values)


def test_shifted_window_and_two_stage():
    r = runtime(threshold_mode="two_stage", late_threshold=0.2)
    assert r.start == Sampling().percent_to_sigma(0.1)
    assert r.end == Sampling().percent_to_sigma(0.9)
    assert r.active(r.start)
    assert not r.active(r.end)
    assert r.threshold(r.split + 1e-4) == 0.08
    assert r.threshold(r.split) == 0.2


def test_same_and_increasing_sigma_invalidates_only_one_stream():
    r = runtime()
    a, b = r.stream("a", 0.9), r.stream("b", 0.9)
    a.tail = b.tail = torch.ones(1, 4, 8)
    assert r.stream("a", 0.9).tail is None
    assert r.streams["b"].tail is not None
    a.tail = torch.ones(1, 4, 8)
    assert r.stream("a", 1.0).tail is None


def test_owned_snapshots_and_image_text_guards():
    r = runtime(metric_stride=1)
    x = torch.ones(2, 10, 8)
    previous = r.sample(x, 8)
    x[1, -1] = 2
    current = r.sample(x, 8)
    assert previous[1].max() == 1
    assert r.difference(current, previous) == 0.5
    assert r.difference(current[:1], previous[:1]) == 0


def test_replay_limit_and_real_refresh():
    r = runtime(metric_stride=1, threshold=0.1)
    s = r.stream("a", 0.9)
    x = torch.ones(1, 10, 8)
    indicator = r.sample(x, 8)
    r.store(s, indicator, x + 2, x)
    s = r.stream("a", 0.8)
    assert torch.equal(r.replay(s, indicator, 0.8, x), x + 2)
    assert r.replay(s, indicator, 0.7, x) is None
    assert s.tail.eq(2).all()
    r.store(s, indicator, x + 3, x)
    assert r.replay(s, indicator, 0.7, x).eq(4).all()


def test_zero_threshold_and_nonfinite_history():
    r = runtime(threshold=0)
    s = r.stream("a", 0.9)
    x = torch.ones(1, 2, 8)
    indicator = r.sample(x, 1)
    r.store(s, indicator, x + 2, x)
    assert r.replay(s, indicator, 0.8, x) is None
    r.store(s, indicator, x * math.inf, x)
    assert s.tail is None
    assert r.fallbacks["nonfinite_history"] == 1


def test_nonfinite_reconstruction_is_not_used():
    r = runtime()
    s = r.stream("a", 0.9)
    x = torch.full((1, 2, 8), 60000, dtype=torch.float16)
    s.tail = x.clone()
    s.indicator = r.sample(x, 1)
    assert r.replay(s, s.indicator, 0.8, x) is None
    assert s.tail is None


def test_lru_budget_and_clear():
    r = runtime(max_cache_mb=1)
    x = torch.ones(1, 512, 512)
    for key in range(3):
        s = r.stream(key, 0.9)
        r.store(s, r.sample(x, 256), x + 1, x)
    assert r.bytes() <= r.budget
    assert r.fallbacks["cache_budget"] >= 1
    r.clear()
    assert r.bytes() == 0


@pytest.mark.parametrize("part", [0, 1])
@pytest.mark.parametrize("ratio", [-1, 0.95])
def test_finite_metric_reduction_overflow_cannot_allow_replay(part, ratio):
    r = runtime()
    previous = [torch.ones(1, 2, 2), torch.ones(1, 2, 2)]
    previous[part] = torch.full((1, 2, 2), 2e38)
    current = [value.clone() for value in previous]
    current[part] *= ratio
    assert all(bool(torch.isfinite(value).all()) for value in previous + current)
    anchor = torch.ones(1, 4, 8)
    stream = r.stream("test", 0.9)
    r.store(stream, tuple(previous), anchor + 1, anchor)
    assert math.isinf(r.difference(tuple(current), tuple(previous)))
    assert r.replay(r.stream("test", 0.8), tuple(current), 0.8, anchor) is None
    assert r.hits == 0


def test_empty_streams_obey_lru_limit_before_history_is_stored():
    r = runtime()
    for index in range(8):
        r.stream(index, 0.9)
    r.stream(0, 0.8)
    r.stream(8, 0.9)
    assert len(r.streams) == 8 and 0 in r.streams and 1 not in r.streams
    for index in range(9, 100):
        r.stream(index, 0.9)
    assert list(r.streams) == list(range(92, 100))
    assert r.bytes() == 0

