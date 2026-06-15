# 🧮 MathFlow — 数学建模竞赛全流程工具库

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> 一站式 Python 工具库，覆盖数学建模竞赛中的评价、预测、优化、图论、仿真全链路。
> 统一 API 设计，开箱即用，让你把时间留给模型创新而非重复造轮子。

## ✨ 特性

- **📦 20+ 经典算法**：AHP、TOPSIS、熵权法、CRITIC、灰色关联、模糊综合评价、灰色预测、ARIMA、指数平滑、线性规划、整数规划、遗传算法、PSO、模拟退火、最短路、最小生成树、网络流、TSP、蒙特卡洛、排队论...
- **🎯 统一 API**：所有模型遵循 `创建 → fit() → 结果` 的一致接口
- **📊 内置可视化**：每个模型都自带绘图方法，论文级图表一键生成
- **📝 结果摘要**：`summary()` 方法输出格式化的计算结果，直接复制到论文
- **🔬 开箱即用**：内置示例数据集，5 行代码跑通完整流程

## 🚀 快速开始

### 安装

```bash
# 基础安装
pip install mathflow

# 完整安装 (含深度学习、高级优化)
pip install "mathflow[full]"

# 开发安装
git clone https://github.com/HyperCharon/mathematical-modeling.git
cd mathematical-modeling
pip install -e ".[dev]"
```

### 5 分钟上手

```python
from mathflow.evaluate import AHP, TOPSIS, EntropyWeight
import numpy as np

# ====== AHP 层次分析法 ======
ahp = AHP()
ahp.set_matrix([
    [1,   3,   5],
    [1/3, 1,   3],
    [1/5, 1/3, 1]
])
ahp.fit()
print(f"AHP 权重: {ahp.weights}")
print(f"一致性比率 CR = {ahp.CR:.4f} ✅" if ahp.is_consistent else "❌ 未通过")

# ====== TOPSIS 综合评价 ======
data = np.array([
    [80, 90, 85, 70],   # 方案1
    [70, 80, 90, 80],   # 方案2
    [90, 85, 80, 75],   # 方案3
])
topsis = TOPSIS(data, weights=[0.3, 0.25, 0.25, 0.2], types=[1, 1, 1, -1])
topsis.fit()
print(f"TOPSIS 排名: {topsis.rankings}")

# ====== 灰色预测 ======
from mathflow.predict import GreyPrediction
gp = GreyPrediction([124, 130, 138, 146, 155, 165])
gp.fit()
print(f"未来3年预测: {gp.predict(steps=3)}")
```

## 📚 模块总览

### 评价类 (`mathflow.evaluate`)

| 模型 | 类名 | 说明 |
|------|------|------|
| 层次分析法 | `AHP` | 主观赋权，支持特征值法/几何平均法/算术平均法 |
| TOPSIS | `TOPSIS` | 逼近理想解排序，支持效益型/成本型指标 |
| 熵权法 | `EntropyWeight` | 基于信息熵的客观赋权 |
| CRITIC | `CRITIC` | 基于对比强度和冲突性的客观赋权 |
| 灰色关联 | `GreyRelationalAnalysis` | 序列几何相似度分析 |
| 模糊综合评价 | `FuzzyEvaluation` | 定性评价定量化，支持多种算子 |
| 秩和比法 | `RSR` | 融合参数与非参数统计的综合评价 |

### 预测类 (`mathflow.predict`)

| 模型 | 类名 | 说明 |
|------|------|------|
| 灰色预测 | `GreyPrediction` | GM(1,1)，小样本指数增长预测 |
| ARIMA | `ARIMAModel` | 自回归积分移动平均，自动选参 |
| 指数平滑 | `ExponentialSmoothing` | 一次/Holt/Holt-Winters 指数平滑 |
| 回归预测 | `RegressionPredict` | 线性/多项式/岭/Lasso 回归 |

### 优化类 (`mathflow.optimize`)

| 模型 | 类名 | 说明 |
|------|------|------|
| 线性规划 | `LinearProgramming` | 可视化可行域，PuLP 求解 |
| 整数规划 | `IntegerProgramming` | 支持纯整数/混合整数/0-1 规划 |
| 遗传算法 | `GeneticAlgorithm` | 通用 GA 框架，实数编码 |
| 粒子群 | `PSO` | 自适应惯性权重 PSO |
| 模拟退火 | `SimulatedAnnealing` | Metropolis 准则，高斯扰动 |

### 图论 (`mathflow.graph`)

| 模型 | 类名 | 说明 |
|------|------|------|
| 最短路径 | `ShortestPath` | Dijkstra / Bellman-Ford / Floyd |
| 最小生成树 | `MinSpanningTree` | Kruskal / Prim |
| 网络流 | `NetworkFlow` | 最大流 (Edmonds-Karp) + 最小割 |
| TSP | `TSPSolver` | 暴力/最近邻/2-opt/遗传算法 |

### 仿真 (`mathflow.simulation`)

| 模型 | 类名 | 说明 |
|------|------|------|
| 蒙特卡洛 | `MonteCarlo` | 积分、π估计、风险分析 |
| 排队论 | `QueueModel` | M/M/1、M/M/c、M/M/1/K |

### 机器学习 (`mathflow.ml`)

| 模型 | 类名 | 说明 |
|------|------|------|
| 聚类分析 | `ClusterAnalysis` | KMeans/DBSCAN/层次聚类，自动选K |
| 降维分析 | `DimensionReduction` | PCA 主成分分析，双标图 |

### 数据处理 (`mathflow.data`)

| 工具 | 类名 | 说明 |
|------|------|------|
| 数据加载 | `DataLoader` | CSV/Excel/JSON/剪贴板，内置示例数据 |
| 数据清洗 | `DataCleaner` | 缺失值/异常值/标准化，一键数据报告 |

## 🎯 数模比赛典型用法

### 评价类问题 (每次比赛必用)

```python
from mathflow.evaluate import AHP, TOPSIS, EntropyWeight
import numpy as np

# 1. 主观权重 (AHP)
ahp = AHP()
ahp.set_matrix([[1, 3], [1/3, 1]])
ahp.fit()
w1 = ahp.weights

# 2. 客观权重 (熵权法)
ew = EntropyWeight(data)
ew.fit()
w2 = ew.weights

# 3. 组合权重
w = 0.6 * w1 + 0.4 * w2

# 4. TOPSIS 排序
topsis = TOPSIS(data, weights=w, types=[1, 1, -1])
topsis.fit()
print(topsis.summary(labels=["方案A", "方案B", "方案C"]))
```

### 预测类问题

```python
from mathflow.predict import GreyPrediction, ARIMAModel

# 小样本 → 灰色预测
gp = GreyPrediction(data)
gp.fit()
forecast = gp.predict(steps=5)

# 大样本 → ARIMA
model = ARIMAModel(data)
model.auto_fit()  # 自动选参
forecast = model.predict(steps=10)
```

### 优化类问题

```python
from mathflow.optimize import LinearProgramming

lp = LinearProgramming()
lp.set_objective([4, 3], sense="max")
lp.add_constraint([2, 1], "<=", 10)
lp.add_constraint([1, 1], "<=", 8)
result = lp.solve()
fig = lp.plot_feasible_region()  # 可行域可视化
```

## 📁 项目结构

```
mathematical-modeling/
├── mathflow/
│   ├── evaluate/      # 评价类模型 (AHP, TOPSIS, 熵权法, CRITIC, GRA, 模糊, RSR)
│   ├── predict/       # 预测类模型 (灰色, ARIMA, 指数平滑, 回归)
│   ├── optimize/      # 优化类模型 (LP, IP, GA, PSO, SA)
│   ├── graph/         # 图论 (最短路, MST, 网络流, TSP)
│   ├── simulation/    # 仿真 (蒙特卡洛, 排队论)
│   ├── ml/            # 机器学习 (聚类, 降维)
│   ├── data/          # 数据处理 (加载, 清洗)
│   ├── viz/           # 可视化 (论文风格)
│   ├── core/          # 核心工具 (配置, 验证)
│   ├── report/        # 报告生成
│   └── templates/     # 赛题模板
├── examples/          # 使用示例
├── tests/             # 测试用例
└── docs/              # 文档
```

## 🤝 参考项目

本项目参考了以下优秀项目，并在其基础上做了统一 API 设计和功能增强：

- [zhanwen/MathModel](https://github.com/zhanwen/MathModel) — 数模论文资源合集
- [datawhalechina/intro-mathmodel](https://github.com/datawhalechina/intro-mathmodel) — Datawhale 数模教程
- [Valdecy/pyDecision](https://github.com/Valdecy/pyDecision) — 多准则决策库
- [HuangCongQing/Algorithms_MathModels](https://github.com/HuangCongQing/Algorithms_MathModels) — 算法与数学模型

## 📄 License

MIT License

---

**如果觉得有用，给个 ⭐ 吧！**
