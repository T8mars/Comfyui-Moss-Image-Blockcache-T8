# 实测验证 / Measured validation — v0.1.0

按用户要求采用有限验证，并已停止扩展跑图。完整 36 组矩阵、同 seed 重复计时和参数敏感性探测未完成，不作为本次交付的完成条件。

- 51 项自动化测试通过，0 failures/errors/skips；包含真实微型原生 MOSS 的跳层、输出头、回退、清理和工作流检查。
- Base、SFT 均在真实 ComfyUI 画布点击运行成功，保存 JSON 和内嵌工作流的 PNG，并重新导入验证。
- 完整权重 SFT 1024² / 50 步零阈值结果与未接插件的原生基线逐像素一致。
- 真实微型原生 MOSS CUDA BF16：GPU/CPU 缓存输出一致；不代表完整权重 CPU 缓存性能。

已经实际检查 18 组成对样本。KSampler 节点中位数加速 **1.218×**，范围 1.197–1.283×；整工作流中位数 1.137×。这是有限 SFT 样本的观测，不能推广为 Base 或其他设备的速度保证。

## 测试条件

ComfyUI 0.42.0 / `8f04fd6ecf4a548937971e840af2a985529fab76`；Python 3.10.10；PyTorch 2.7.0+cu128；cuda:0 NVIDIA GeForce RTX 5090 Laptop GPU : cudaMallocAsync；原生 PyTorch attention。
使用本地已有原生 BF16 Base/SFT、Qwen3.5-9B 文本编码器和官方 MOSS VAE，无模型下载或转换。1024×1024，50 步，Euler/simple，CFG3，negative skip22；原生 shift5 / multiplier1，不接 ModelSamplingSD3。缓存阈值0.05，窗口0.10–0.90，最多连续命中1次。
测试服务关闭执行结果缓存、dynamic VRAM 和 Comfy compiler。KSampler 节点计时包含 setup 和模型加载；整流程另含编码、解码和保存。基线先于缓存运行，未做随机顺序的同 seed 重复测速。

| 模型 | 场景 | 已审对数 | KSampler 中位数加速 | 整流程中位数加速 |
|---|---|---:|---:|---:|
| SFT | anime | 3 | 1.211× | 1.112× |
| SFT | chinese | 3 | 1.226× | 1.151× |
| SFT | english | 3 | 1.222× | 1.135× |
| SFT | layout | 3 | 1.211× | 1.164× |
| SFT | people | 3 | 1.214× | 1.124× |
| SFT | photo | 3 | 1.221× | 1.121× |

## 已观察到的限制

近似缓存会改变面部、头发、服装配件、手部姿态、杯子比例、背景灯具和植物比例。已审中文招牌主要文字保留；部分英文原生基线已有换行偏差，缓存保留该偏差。布局样本保留主体位置，轮廓和高光存在细微变化。逐组记录、prompt ID 和图片哈希见 [validation/summary.json](validation/summary.json)。像素指标不能证明文字准确或美学质量。
本次验收范围为原生 dense MOSS Base/SFT BF16、普通文生图及 JSON Layout、eager PyTorch 后端和本机 GPU。没有其他 GPU、量化/FP16、编译、低显存 offload、参考图编辑或 HF pipeline 一致性的实测结论。
