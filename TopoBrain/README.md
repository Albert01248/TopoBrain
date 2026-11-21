# TopoBrain

冻结 CodeBrain TFDual-Tokenizer 的 EEG 分类实验代码，包含代码提取、空间图传播、局部时间读出、缺导评估和硬件计时。

## 目录

```text
topobrain/          模型、图构造、数据校验、训练与命令行接口
configs/            五组实验配置
scripts/            数据准备、实验运行与诊断脚本
tests/              单元测试
requirements.txt   Python 依赖
```

## 安装

使用 Python 3.12，在仓库根目录执行。Windows PowerShell：

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

CPU 环境将 PyTorch 索引改为 `https://download.pytorch.org/whl/cpu`。
Linux 对应使用 `python3.12 -m venv .venv` 和 `.venv/bin/python`。

## 测试与快速运行

单元测试和合成输入检查无需下载 EEG 数据或模型权重：

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
& .\.venv\Scripts\python.exe scripts/make_synthetic_fixture.py --output-dir data/synthetic_example
& .\.venv\Scripts\python.exe scripts/run_five_experiments.py --cache data/synthetic_example/synthetic_codes.npz --split data/synthetic_example/synthetic_split.json --output results/synthetic_example --smoke
```

`--smoke` 使用1轮训练和首个随机种子，运行27个配置。合成输入只检查代码流程。
每次运行使用新的输出目录。

## 准备 FACED 输入

模型与数据来源见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

```powershell
& .\.venv\Scripts\python.exe scripts/fetch_assets.py
& .\.venv\Scripts\python.exe scripts/prepare_faced_pilot.py --output data/faced_pilot/raw.npz --subjects sub-000 sub-001 sub-002 --windows-per-class 2 --download
& .\.venv\Scripts\python.exe -m topobrain extract --raw data/faced_pilot/raw.npz --repo vendor/CodeBrain --checkpoint weights/CodeBrain_Tokenizer.pth --output data/faced_pilot/codes.npz --device auto --batch-size 1
& .\.venv\Scripts\python.exe -m topobrain inspect --cache data/faced_pilot/codes.npz --split data/faced_pilot/split.json
```

数据准备脚本固定使用3名受试者，生成互斥的训练/验证/测试划分。
预处理包括 BDF 物理标尺转换、0.3--45 Hz 滤波、200 Hz 重采样和微伏/100缩放。
图坐标使用标准电极模板；Tokenizer 通道位置约定记录在缓存元信息中。

## 五组实验

| 实验 | 配置 | 实现模块 |
|---|---|---|
| 1：五种图结构对照 | [experiment_1.json](configs/experiment_1.json) | graphs.py、model.py |
| 2：缺导评估 | [experiment_2.json](configs/experiment_2.json) | experiments.py、tokenizer.py |
| 3：邻居数与图深度 | [experiment_3.json](configs/experiment_3.json) | graphs.py、experiments.py |
| 4：时域/频域及空间/时间上下文 | [experiment_4.json](configs/experiment_4.json) | model.py、experiments.py |
| 5：读出头参数与硬件计时 | [experiment_5.json](configs/experiment_5.json) | experiments.py |

实现模块位于 `topobrain/`。统一运行入口：

```powershell
& .\.venv\Scripts\python.exe scripts/run_five_experiments.py --cache data/faced_pilot/codes.npz --split data/faced_pilot/split.json --output results/faced_run --neighbors 4 --smoke
```

去掉 `--smoke` 使用常规预算，默认最多30轮、3个随机种子。单独运行一组：

```powershell
& .\.venv\Scripts\python.exe -m topobrain run --experiment configs/experiment_1.json --cache data/faced_pilot/codes.npz --split data/faced_pilot/split.json --output results/experiment1 --smoke
```

单实验使用配置原值 `k=2`；统一脚本的 `--neighbors 4` 覆盖相应图配置。
实验3扫描配置中的邻居数2和4。各训练目录保存 `manifest.json`、`metrics.json`、
`predictions.npz` 和 `best.pt`，模型按验证集 macro-F1 选择。

### 原始输入缺导

缓存 mask 评估分类头；输入阶段缺导需要删除通道并重新提取代码。
下面的通道名单适用于上述 FACED 模板：

```powershell
& .\.venv\Scripts\python.exe -m topobrain extract --raw data/faced_pilot/raw.npz --repo vendor/CodeBrain --checkpoint weights/CodeBrain_Tokenizer.pth --output data/faced_pilot/codes_raw_missing_left.npz --device auto --batch-size 1 --drop-channels Fp1,F3,F7,FC1,FC5,C3,T3,A1,CP1,CP5,P3,T5,PO3,O1
& .\.venv\Scripts\python.exe scripts/run_five_experiments.py --cache data/faced_pilot/codes.npz --split data/faced_pilot/split.json --raw-missing-cache data/faced_pilot/codes_raw_missing_left.npz --output results/faced_missing_run --neighbors 4 --smoke
```

### 表示诊断与计时

```powershell
& .\.venv\Scripts\python.exe scripts/validate_tokenizer.py --raw data/faced_pilot/raw.npz --cache data/faced_pilot/codes.npz --repo vendor/CodeBrain --checkpoint weights/CodeBrain_Tokenizer.pth --output results/tokenizer_check.json
& .\.venv\Scripts\python.exe scripts/diagnose_branch_features.py --raw data/faced_pilot/raw.npz --repo vendor/CodeBrain --checkpoint weights/CodeBrain_Tokenizer.pth --output results/branch_features.json --max-samples 9 --device auto
& .\.venv\Scripts\python.exe scripts/check_train_fit.py --cache data/faced_pilot/codes.npz --split data/faced_pilot/split.json --output results/train_fit.json --steps 200 --lr 0.003 --device auto
& .\.venv\Scripts\python.exe scripts/profile_pipeline.py --raw data/faced_pilot/raw.npz --cache data/faced_pilot/codes.npz --run-dir results/faced_run/experiment_01/geometry_seed42 --repo vendor/CodeBrain --checkpoint weights/CodeBrain_Tokenizer.pth --output results/prepared_profile.json --device auto --warmup 2 --iterations 5 --batch-size 1
```

`check_train_fit.py` 只拟合训练样本。`profile_pipeline.py` 计时已预处理内存输入到设备输出，
包含编码器和量化，排除文件读取、滤波、重采样和模型初始化。

## 模型与数据约定

- 原始预处理输入为 `[N,C,P,200]`；离散代码为 `codes_t/f[N,C,P]`。
- 缓存包含码本、标签、受试者ID、通道ID、坐标和 JSON 来源元信息，禁止 pickle。
- 训练、验证和测试按受试者互斥；训练标签须覆盖所有类别。
- 缺导使用布尔 `valid_channel_mask[N,C]`；代码0是合法ID。
- 空间对照保留层参数、改变图边；时间对照保留 attention/FFN、改变上下文窗口。
- 时间层使用密集 attention 加局部 mask，处理 patch 之间的上下文。
- Tokenizer 和码本冻结；训练前应检查代码使用情况。
