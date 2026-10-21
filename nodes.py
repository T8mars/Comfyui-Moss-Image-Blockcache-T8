from comfy_api.latest import ComfyExtension, io

from .cache import CacheConfig
from .runtime import install


class MOSSImageBlockCacheT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MOSSImageBlockCacheT8", display_name="MOSS Image Block Cache (T8)", category="T8/MOSS-Image",
            description="Native MOSS-Image first-block guarded residual reuse. Higher thresholds can change text, layout and details. External block patches run without caching.",
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input("enabled", default=True),
                io.Float.Input("threshold", default=0.08, min=0.0, max=1.0, step=0.005,
                               tooltip="0 disables residual reuse. Compare with the same native workflow without caching."),
                io.Float.Input("start_percent", default=0.10, min=0.0, max=1.0, step=0.01),
                io.Float.Input("end_percent", default=0.90, min=0.0, max=1.0, step=0.01),
                io.Int.Input("max_consecutive_hits", default=1, min=1, max=10),
                io.Combo.Input("cache_device", options=["gpu", "cpu"], default="gpu"),
                io.Int.Input("max_cache_mb", default=512, min=16, max=32768, advanced=True),
                io.Int.Input("metric_stride", default=4, min=1, max=32, advanced=True),
                io.Combo.Input("threshold_mode", options=["constant", "two_stage"], default="constant", advanced=True),
                io.Float.Input("split_ratio", default=0.30, min=0.0, max=1.0, step=0.01, advanced=True),
                io.Float.Input("late_threshold", default=0.14, min=0.0, max=1.0, step=0.005, advanced=True),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, enabled=True, threshold=0.08, start_percent=0.10, end_percent=0.90,
                max_consecutive_hits=1, cache_device="gpu", max_cache_mb=512, metric_stride=4,
                threshold_mode="constant", split_ratio=0.30, late_threshold=0.14):
        config = CacheConfig(threshold, start_percent, end_percent, max_consecutive_hits, cache_device,
                             metric_stride, max_cache_mb, threshold_mode, split_ratio, late_threshold)
        return io.NodeOutput(install(model, config, enabled))


class MOSSImageBlockCacheT8Extension(ComfyExtension):
    async def get_node_list(self):
        return [MOSSImageBlockCacheT8]

