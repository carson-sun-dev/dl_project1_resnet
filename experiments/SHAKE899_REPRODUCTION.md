# Shaohan Yang 的 89.9% Shake-Shake 方案

贡献者：Shaohan Yang（GitHub：SHAOHANYANG）。

本分支交付公榜 0.899 的单模型训练与推理代码、实际配置和优化报告。分支不包含比赛数据、划分 ID、模型权重、提交预测或云服务凭据。完整优化历程见 [优化报告](SHAOHAN_OPTIMIZATION_REPORT.md)。报告中的在线蒸馏和其他历史实验仅作为结果对照，本分支不包含这些旁支代码。

## 实际配方

Shake-Shake-26 2×40d，4,561,354 参数，50k 许可训练数据，随机初始化，seed=2026，1000 轮，batch=128，SGD Nesterov momentum=0.9、weight decay=1e-4；5 轮 warmup 到 0.2，余弦降至 0。增强为随机裁剪、水平翻转、CIFAR10 AutoAugment NEAREST、归一化、Cutout16。训练 BF16，最终推理 FP32 原图与翻转 logits 平均。没有独立验证集，固定导出最后一轮。

`configs/shake_full1000_s2026.json` 是实际完成运行的配置快照。`shake_refit` 的入口由经过核验的源配方构建设置，并使用 `--epochs 1000` 指定总轮数；它不接受 `--config`。不要把配置快照传给一般的 `src.train`。

## 环境与私有输入

建议使用 Python 3.10，并安装适合本机 CUDA 的 PyTorch 2.8.0 / torchvision 0.23.0。其余依赖参见 `requirements-shake899.txt`；该文件记录 A 的 NumPy/Pillow 版本及打包依赖。CUDA 训练环境记录为 CUDA 12.8、cuDNN 91002。

先在本地放置合法取得的 `data/train_images`、`data/test_images`、`train_labels.csv`、`test.csv`、`sample_submission.csv`。还需要通过组内私有渠道取得 `runs/shake_aa_s2026/` 中的 `best.pt`、`config.json`、`summary.json`、`split.json`、`validation/metrics.json`。

该源运行记录用于严格检查数据划分、来源和已经选择的配方；scratch 模式不使用源权重初始化。这不是一个完全不依赖私有运行材料的训练入口。全量原运行所需的源权重哈希已固定在代码中。自己重新训练 45k 模型可能无法逐位产生同一权重；此时需为新的实验建立匹配的来源记录与检查，不应删除检查或将其冒充原实验。

## 运行命令

从仓库根目录执行；使用新目录保存复现实验，不覆盖原结果。

```powershell
python -m src.shake_refit --mode scratch --epochs 1000 --run-dir runs/shake_full1000_reproduction --device cuda
python -m src.predict --checkpoint runs/shake_full1000_reproduction/final.pt --output submissions/shake_full1000_reproduction/submission.csv --device cuda
```

恢复同一次实验时，在原训练命令后追加 `--resume`，保持轮数、配置、精度及数据不变。推理模型内部已包含水平翻转，不再额外套用镜像推理。

独立复现 45k 选择实验可使用 `python -m src.shake_train --config configs/shake_aa_s2026.json --device cuda`，但需先私下取得匹配的划分文件；该命令不会直接生成 0.899 全量版。

## 验证与发布范围

```powershell
python -m unittest discover -s tests -v
python -m src.models --config configs/shake_aa_s2026.json
```

测试用合成图片覆盖双分支梯度、投影捷径、镜像推理、训练恢复、数据划分变化拒绝、全量训练导出和提交 ID 对齐。为独立打包，合成数据夹具不再依赖 SAM 测试；通用训练锁改为跨平台 FileLock，模型注册和推理元数据增加 Shake-Shake 支持。Shake-Shake 核心训练与网络文件保留原实现。运行记录中的源代码哈希属于原训练工作区，不能当成本分支每个共享模块的哈希；跨硬件也不保证逐位一致。

原最终权重 SHA256：`324bd810a33d158ff6487fce51a02b12a9076cab65d38fef379b9420adb38c17`。

原提交 SHA256：`cb238fdce299420bfcf504aa512226e83783649f4db4f389abec79cc0fd6d69f`。

Public Score 0.899 来自提交后的组内确认，Private Score 未知。
