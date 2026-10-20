# ComfyUI MOSS Image BlockCache T8

**简体中文** · [English](README_EN.md)

适用于 **ComfyUI 官方原生 MOSS-Image Base / SFT** 的实验性 BlockCache 加速节点。通过首块变化检测与尾部残差复用减少重复计算，保留原生模型加载、采样和 VAE，直接使用本地已有模型，无需新增推理依赖。

[Comfy Registry](https://registry.comfy.org/publishers/t8star/nodes/moss-image-blockcache-t8) · [实测报告](BENCHMARKS.md) · [问题反馈](https://github.com/T8mars/Comfyui-Moss-Image-Blockcache-T8/issues)

## 安装

需要 **ComfyUI 0.42.0 或更新版本，且已支持原生 MOSS-Image**。

在 ComfyUI Manager 中搜索 **MOSS Image BlockCache T8**（节点包 ID：`moss-image-blockcache-t8`），安装后重启。也可在 `ComfyUI/custom_nodes` 中手动安装：

```bash
git clone https://github.com/T8mars/Comfyui-Moss-Image-Blockcache-T8.git comfyui-moss-image-blockcache-T8
```

插件不下载、转换或修改模型。使用自定义节点白名单启动时，将安装目录名加入 `--whitelist-custom-nodes`。

## 工作流

Base、SFT 均已在真实 ComfyUI 画布运行，JSON 和生成图片内的工作流已重新导入验证。

| 模型 | 画布工作流 | 内嵌工作流的图片 |
|---|---|---|
| Base | [JSON](workflows/MOSS_T8_BASE_1024_BlockCache.json) | [PNG](workflows/MOSS_T8_BASE_1024_BlockCache.png) |
| SFT | [JSON](workflows/MOSS_T8_SFT_1024_BlockCache.json) | [PNG](workflows/MOSS_T8_SFT_1024_BlockCache.png) |

拖入 JSON 或 PNG，在加载器中选择本地原生 DiT、`type=moss_image` 的文本编码器和匹配 VAE。示例预选 T8 BF16 文件名，可按本机文件修改；`*-api.json` 是 API 格式。

```text
Load Diffusion Model → MOSS Image Block Cache (T8)
    → SkipLayerGuidanceDiTSimple（负分支 skip22）→ KSampler
```

节点位于 `T8/MOSS-Image` 分类，输入、输出均为标准 `MODEL`。示例为 1024×1024、50 步、Euler / simple、CFG 3、seed 42。

**保留 MOSS 原生 shift=5、timestep multiplier=1。不要添加通用 `ModelSamplingSD3`，它会改变时间尺度，导致错误结果。**

## 参数

| 参数 | 默认值与说明 |
|---|---|
| `enabled` | 关闭后完整透传，便于与原生基线对照 |
| `threshold` | `0.05`；越大越容易复用，也越可能改变文字、人物和细节 |
| `start_percent` / `end_percent` | `0.10 / 0.90`；区间外完整计算 |
| `max_consecutive_hits` | `1`；命中后强制刷新 |
| `cache_device` | `gpu`；`cpu` 用传输开销换取较低的缓存显存占用 |
| `max_cache_mb` / `metric_stride` | `512 MiB / 4`；限制保留缓存与指标采样间隔 |
| `threshold_mode` | `constant`；可选 `two_stage`，配合 `split_ratio`、`late_threshold` 调整后段阈值 |

固定阈值模式下 `threshold=0` 禁用缓存；两段模式还需将 `late_threshold` 置零，或直接关闭 `enabled`。缓存预算不等于整条工作流的显存上限。

## 实测与兼容

已验证原生 dense BF16 Base/SFT 文生图及 JSON Layout。完整权重 SFT 零阈值输出与原生基线逐像素一致；实测条件、有限样本耗时和逐组差异见 [BENCHMARKS.md](BENCHMARKS.md)。

`0.1.3` 修正模型克隆对象识别，原生多 GPU 配置回退完整计算，并补齐手动安装包的检查附件。画布与测速记录对应 `0.1.0`；本补丁经原生 CPU 回归验证，见 [检查记录](validation/review-0.1.3.json)。

缓存属于近似加速，可能改变面部、文字、手部、布局和细节。区域条件、Control、动态权重 hooks、外部 block patch 等不支持组合回退完整计算；原生负分支 skip22 保留，正分支仍可加速。请勿叠加其他缓存插件。量化、参考图编辑、编译、低显存 offload 和其他 GPU / dtype 不在本版实测范围。

## 社媒与资源

[B站](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [Hugging Face](https://huggingface.co/t8star)

[API](https://api.seedance.nz/sign-up?aff=5f4w) · [免费画廊](https://www.openzhenzhen.com) · [在线 AI 应用](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI 整合包](https://pan.quark.cn/s/264edb7e36bd)

## 许可与致谢

[Apache-2.0](LICENSE)。参考 [Qwen-Image-2.1 BlockCache T8](https://github.com/T8mars/Comfyui-Qwen-Image-2.1-BlockCache-T8)，针对 [MOSS-Image](https://github.com/OpenMOSS/MOSS-Image) 与 [ComfyUI](https://github.com/Comfy-Org/ComfyUI) 原生实现适配。模型权重不随插件分发，详见 [NOTICE](NOTICE.md)。
