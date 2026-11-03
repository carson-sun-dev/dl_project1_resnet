# 队友使用指南：CIFAR-10 ResNet 项目

这是一套训练代码、三份正式模型配置。每次只运行一个模型，比较同一验证集上的结果，最终选一个模型生成 Kaggle 提交文件。先跑 Baseline，再跑 Wide；Deep 可按时间决定。本指南不代表已有正式实验成绩。

## 1. 第一次拿到项目

支持 macOS / Linux，Python 3.9–3.12。Windows 请使用 WSL2（训练使用 Unix 文件锁）。以下命令在终端运行；在 JupyterLab 中打开 Terminal，不要直接粘贴进 Python 单元格。

```bash
git clone https://github.com/carson-sun-dev/dl_project1_resnet.git
cd dl_project1_resnet
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip freeze > requirements-installed.txt
```

已经有项目的同学进入自己的项目目录即可。所有后续命令都从项目根目录运行。仓库不会自动下载数据，也不包含训练权重。

把老师提供的文件放成以下结构，注意不要多套一层 `data/data/`：

```text
data/
├── train_images/           # 50,000 张 RGB PNG
├── test_images/            # 2,000 张无标签评估图
├── train_labels.csv        # id,filename,label
├── test.csv                # id,filename
└── sample_submission.csv   # id,label
```

检查设备：

```bash
.venv/bin/python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available())"
```

Apple Silicon Mac 使用 `--device mps`；分配到 NVIDIA GPU 的计算节点使用 `--device cuda`。自动模式依次选择 CUDA、MPS、CPU；正式训练建议显式指定设备，避免不知情地落到 CPU。CUDA 使用混合精度；MPS 使用 FP32。HPC 环境需先确认安装的 PyTorch 能识别 GPU，代码不申请计算资源。

## 2. 检查数据，生成固定划分

```bash
.venv/bin/python -m src.data audit --data-dir data --verify-images
.venv/bin/python -m src.data prepare --data-dir data --output splits/seed2026.json
```

预计得到 45,000 张训练图、5,000 张验证图（每类验证图 500 张）。均值、标准差只从训练部分计算。相同划分文件存在时会检查后复用。

队友必须使用相同数据、划分和预处理，才能公平比较模型。建议队内共享同一个 `splits/seed2026.json`；该文件默认不上传公开 GitHub。可以用下面命令比对 SHA-256：

```bash
.venv/bin/python -c "from src.utils import file_hash; print(file_hash('splits/seed2026.json'))"
```

隐藏评估图只用于最后预测，不参与训练、统计量拟合或验证集模型选择。`sample_submission.csv` 的零是格式占位符，不是真实标签。

## 3. 选择模型

| 配置文件 | 通道数 | 每阶段残差块 | 可训练参数 |
|---|---|---|---:|
| `configs/baseline.json` | 32 / 64 / 128 | 3 / 3 / 3 | 1,084,586 |
| `configs/wide.json` | 64 / 128 / 256 | 3 / 3 / 3 | 4,327,754 |
| `configs/deep.json` | 48 / 96 / 192 | 5 / 5 / 5 | 4,180,858 |

三者都从随机权重训练，有残差连接，构建时会检查 500 万参数上限。正式默认：200 epochs、batch size 128、SGD、momentum 0.9、weight decay 0.0005、学习率 0.1、5 epoch warmup 后 cosine decay。增强为随机裁剪、水平翻转。

`configs/smoke.json` 是少量 batch 的流程检查，不是第四个正式候选。它的准确率和预测不能作为竞赛成果。

## 4. 开始运行（准备好后再执行）

首次使用可先做流程检查：

```bash
.venv/bin/python -m src.workflow --config configs/smoke.json --device mps
```

正式训练使用统一入口；一次只选下面一条，不需要全部执行：

```bash
.venv/bin/python -m src.workflow --config configs/baseline.json --device mps
.venv/bin/python -m src.workflow --config configs/wide.json --device mps
.venv/bin/python -m src.workflow --config configs/deep.json --device mps
```

在 NVIDIA GPU 上把 `mps` 替换成 `cuda`。程序会训练、保存权重和日志、绘制曲线，并在训练完成后评估最佳模型、输出混淆矩阵。默认不会对隐藏评估集预测或上传 Kaggle。

Mac 长时间训练可用下面的命令在进程运行期间阻止系统空闲睡眠，保持接电和开盖：

```bash
caffeinate -i .venv/bin/python -m src.workflow --config configs/baseline.json --device mps
```

每个完成的 epoch 打印训练 loss、验证 loss、验证 accuracy、训练耗时、当前最佳 epoch。一个 epoch 内控制台暂时无新行属正常现象；可通过设备活动和输出文件判断进程状态。

多个队友分工运行不同配置即可。同一机器建议依次运行；每个实验必须使用独立输出目录。例如：

```bash
.venv/bin/python -m src.workflow --config configs/wide.json --device cuda --run-dir runs/wide_alice_s2026
```

修改超参数时复制 JSON 为新配置，并设置新的 `run_dir`。保留旧配置和旧实验产物，不要把多个实验写进同一目录。

## 5. 中断与恢复

每个 epoch 结束保存 `last.pt`。中途断开时，未完成的那个 epoch 需要重跑。恢复命令：

```bash
.venv/bin/python -m src.workflow --config configs/baseline.json --device mps --resume
```

自定义过 `--run-dir`、`--data-dir`、`--split-file` 时，恢复时继续提供同样的参数。保持原模型、训练超参数和源代码版本。跨设备/软件版本恢复不保证逐位相同。

想先试 3 个 epoch 而保留 200 epoch 学习率计划：

```bash
.venv/bin/python -m src.workflow --config configs/baseline.json --device mps --stop-after-epoch 3
.venv/bin/python -m src.workflow --config configs/baseline.json --device mps --resume
```

暂停后会生成部分日志和曲线，`summary.json` 的 `complete` 为 `false`；完整验证分析在计划训练完成后生成。不要把部分实验当作已经完成的 200 epoch 实验。

训练完成但画图/评估阶段报错时，修复原因后用同样的 `--resume` 命令。若 `last.pt` 已到计划最后一轮，程序只恢复模型和历史、补做导出，不再训练新 epoch。

`best.pt` 用于验证和预测；`last.pt` 用于恢复训练，二者不能互换。只加载自己或队友可信的 checkpoint。

## 6. 运行后应该获得哪些文件

以 Baseline 为例：

```text
runs/baseline_s2026/
├── config.json               # 实际使用的配置
├── split.json                # 该次实验的固定划分与训练集归一化参数
├── environment.json          # Python、Torch、设备、参数量、源代码哈希、Git commit
├── best.pt                   # 验证 accuracy 最佳模型；相同 accuracy 时比较 loss
├── last.pt                   # 最近完成的 epoch 和完整恢复状态
├── metrics.json              # 每个 epoch 的训练/验证指标
├── metrics.csv               # 同一组指标，便于表格比较
├── summary.json              # 是否完成、最佳 accuracy/epoch、参数量、耗时
├── curves.png                # 训练/验证 loss 与 accuracy 曲线
├── artifacts.json            # 本次流程成功导出的文件大小和 SHA-256
└── validation/
    ├── metrics.json          # 最佳 checkpoint 的完整验证集 FP32 重评估
    ├── per_class.csv         # 各类样本数、准确率、precision、recall、F1
    ├── predictions.csv       # 验证图 ID、真实标签、预测标签、是否正确
    ├── confusion_matrix.csv  # 行=真实类别，列=预测类别，数值=图片数
    └── confusion_matrix.png  # 混淆矩阵图
```

`metrics.json` 与 checkpoint 每轮更新；`metrics.csv`、`summary.json` 在训练正常退出或恢复后导出。`artifacts.json` 仅在整个 workflow 成功完成导出时刷新。意外中断后先恢复，再判断这些文件是否最新。

所有 JSON/CSV 中 accuracy 为 0–1，例如 `0.93` 表示 93%；图上显示百分数。训练 accuracy 使用增强后的训练图，验证 accuracy 使用无随机增强的验证图。

比较模型主要看 `summary.json` 的 `best_val_accuracy`，同时观察曲线是否过拟合。`validation/metrics.json` 使用 FP32 重新评估，因此与 CUDA 混合精度训练时保存的值可能略有不同；记录比较口径。验证指标不是 Kaggle 隐藏集准确率。

训练时间包含训练数据读取和更新，不含验证。`compute_seconds_with_validation` 包含验证及部分保存开销，不是从进程启动到退出的完整墙钟时间。MPS/CPU 的峰值 CUDA 显存字段为 `null`，不是零显存。

也可以单独补做图与验证分析，不启动训练：

```bash
.venv/bin/python -m src.plot --run-dir runs/baseline_s2026
.venv/bin/python -m src.evaluate --run-dir runs/baseline_s2026 --data-dir data --device mps
```

单独验证会覆盖该 run 的 `validation/` 派生分析文件，但不会改动模型权重。它会验证划分与 checkpoint 匹配，且只读取原验证集。

## 7. 选定模型，生成 Kaggle 提交

先根据固定验证集选定一个候选。下面以 Wide 为例，实际换成选中的目录：

```bash
.venv/bin/python -m src.predict --checkpoint runs/wide_s2026/best.pt --data-dir data --output submissions/wide_s2026/submission.csv --device mps
.venv/bin/python -m src.submission --file submissions/wide_s2026/submission.csv --sample data/sample_submission.csv
```

得到：

```text
submissions/wide_s2026/
├── submission.csv
└── submission.metadata.json
```

正式数据对应 **2,000 行预测加一行表头**，表头严格为 `id,label`，label 是整数 0–9，ID 顺序与 `sample_submission.csv` 一致。导出时自动校验，不添加 pandas 索引列。输出文件存在时会拒绝覆盖，新的候选请换目录。

`submission.metadata.json` 记录模型与 CSV 哈希、模型参数量、checkpoint epoch。预测使用单模型 FP32，不做集成或测试时增强。smoke 权重默认禁止导出。

如果已经确定要在某次完整训练后直接生成 CSV，可以显式加参数：

```bash
.venv/bin/python -m src.workflow --config configs/wide.json --device mps --submission-output submissions/wide_s2026/submission.csv
```

这个命令也会真正训练；已有完成实验请用前面的 `src.predict`。workflow 不允许 smoke 或计划内提前停止的实验自动生成提交。它不会登录或上传 Kaggle。

在 Kaggle 的 File Upload 页面只上传 `submission.csv`。团队安排一个人负责上传，避免重复消耗共享提交次数。公开榜反馈与最终私有榜分数可能不同。

## 8. 队内保存与协作

- GitHub 保存代码、配置、说明和真实实验摘要；数据、权重、运行目录、提交 CSV、环境与凭证已被 `.gitignore` 排除。
- 每人从最新代码创建自己的分支，修改后开 PR。不要在正在训练的目录切换源代码版本。
- 队内另行保存完整 `runs/<实验名>/`，尤其是 `best.pt`、`last.pt`、配置、划分和日志。Git push 不会备份这些权重。
- 实验完成后手工填写 `experiments/results.csv`：实验名、Git commit、配置、seed、参数量、验证 accuracy、最佳 epoch、耗时、checkpoint SHA-256。未提交 Kaggle 时 `public_score` 留空，不能猜测填写。
- 公开仓库的分享要求见 `CONTRIBUTING.md`；报告后续再整理，现在先保留真实记录。

## 9. 常见问题

| 现象 | 处理 |
|---|---|
| `No module named torch` | 使用项目 `.venv/bin/python`，并在此虚拟环境安装 requirements |
| MPS / CUDA unavailable | 先检查设备输出与当前解释器；HPC 需分配 GPU 计算节点 |
| 找不到 CSV 或图片 | 确认当前目录是项目根目录，检查是否多了一层 data |
| `Run exists` | 继续原实验加 `--resume`；新实验用新 `--run-dir` |
| `No last.pt to resume` | 尚未完成第一个 epoch；选择新的 run 目录重新开始 |
| `Resume config/split differs` | 使用原配置和原划分；不能借 resume 改实验定义 |
| 内存不足 | 新配置减小 batch size，使用新 run 目录；先关闭其他重负载程序 |
| Submission exists | 换新的提交子目录，保留旧文件和哈希记录 |
| 作业被 HPC 回收 | 重新获得计算资源，从持久保存的 last.pt 恢复 |

## 10. 本次交付的验证范围

按要求，本次只编写代码和文档，做静态语法、配置和差异检查；不启动训练、推理或单元测试。原骨架此前的测试记录见 `VALIDATION.md`。新增 workflow 和验证分析尚未经过运行验证，不应把“代码已写好”当作“正式实验已通过”。

队友准备开始运行时，可先执行下面的合成数据测试，再执行 smoke 流程：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

测试会在临时目录做极小训练，因此本次没有执行。正式准确率、曲线、混淆矩阵和 submission 必须等实际运行后才能得到。
