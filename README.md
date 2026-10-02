# 几何约束下无线电干扰源的分层学习定位与清除

无线电干扰源定位与清除的 Python 实现，覆盖四问求解、几何约束、分层 PPO 训练、策略评估和覆盖巡检对照算法。第三、四问以候选宏动作结合保守几何约束进行搜索，统一入口默认加载训练种子 0 的策略。

本仓库为代码整理版，包含源码、训练配置和推理/比较所需的模型权重。论文 PDF、原始结果数据、复现输出、训练日志及最后一轮训练快照不随仓库分发。

## 目录

| 路径 | 用途 |
| --- | --- |
| `main.py` | 四问统一求解入口 |
| `static_geometry/` | 第一、二问几何交会、直径计算、测点优化和数值校核 |
| `omnidirectional/` | 第三问全向源定位：几何、仿真、环境、策略、训练和评估 |
| `mixed_sources/` | 第四问混合源定位：信念更新、覆盖设计、仿真、训练和评估 |
| `baselines/` | 第三、四问覆盖巡检与最小二乘定位对照算法 |
| `*/models/seed{0,1,2}/` | 各训练种子的 `best_ppo.pt`、`bc.pt` 和 `config.json` |
| `requirements.txt` | 运行、训练与数值验证依赖 |

## 安装

推荐 Python 3.11，CPU 即可运行。在仓库根目录执行：

```bash
python -m venv .venv
```

Windows PowerShell 激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS 激活环境：

```bash
source .venv/bin/activate
```

安装依赖：

```bash
python -m pip install -r requirements.txt
```

随附模型原训练环境采用 PyTorch 2.7.1 CPU；依赖范围见 `requirements.txt`。

## 四问求解

```bash
python main.py --question 1
python main.py --question 2
python main.py --question 3 --seed 40000
python main.py --question 4 --seed 140000
```

结果保存至仓库内的 `outputs/question_1.json` 至 `outputs/question_4.json`，也可用 `--output` 指定位置：

```bash
python main.py --question 3 --seed 40000 --output outputs/q3_seed40000.json
```

第三、四问输出清除数、平均定位清除时间、程序运行时间和总虚拟时间；JSON 还保存宏动作及仿真请求记录。求解结果不导出绘图用的干扰源真值坐标。

## 测试与数值校核

```bash
python -m unittest omnidirectional.test_solver mixed_sources.test_solver
python -m static_geometry.checks
```

测试覆盖几何边界、测量误差、通道与清除语义、策略动作掩码及困难布局。数值校核报告保存至 `outputs/static_geometry_checks.json`。

## 重新训练

下列命令把新模型写入 `reproduced/`：

```bash
python -m omnidirectional.train --seed 0 --steps 16384 --validation 16 --validate-every 8 --output reproduced/omnidirectional/seed0
python -m mixed_sources.train --seed 0 --steps 32768 --validation 24 --validate-every 16 --output reproduced/mixed_sources/seed0
```

将 `--seed` 改为 `1`、`2` 可训练另外两组策略。各组原训练参数和场景种子范围见随附的 `config.json`。重新训练结果可能因软件版本、硬件和数值计算差异而变化。

## 策略评估与对照比较

比较随附的三个训练种子模型：

```bash
python -m omnidirectional.evaluate --models omnidirectional/models --output reproduced/omnidirectional_results --n 200 --stress-n 20
python -m mixed_sources.evaluate --models mixed_sources/models --output reproduced/mixed_sources_results --n 200 --stress-n 20
```

比较重新训练的模型时，将 `--models` 分别换为 `reproduced/omnidirectional` 和 `reproduced/mixed_sources`。评估还会加载所选模型所在目录的 `bc.pt`，因此随附行为克隆权重与 PPO 权重共同保留。

## 整理说明

源码来自“几何约束下无线电干扰源的分层学习定位与清除/代码”。模块直接放在仓库根目录，默认输出路径统一到仓库内的 `outputs/`；算法和随附模型权重保持原内容。运行输出、缓存和虚拟环境由 `.gitignore` 排除。
