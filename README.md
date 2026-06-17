# 🧮 MathFlow — 数学建模竞赛全流程工具库

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-230%20passed-brightgreen.svg)](#测试)
[![Coverage](https://img.shields.io/badge/真题验证-12%20道-orange.svg)](#真题实战验证)

> 一站式 Python 工具库，覆盖数学建模竞赛中的 **评价、预测、优化、图论、仿真、ODE、统计、ML、博弈论、动态规划、时间序列、模糊逻辑、马尔可夫、灰色系统、小波分析、论文写作** 全链路。
>
> 统一 API 设计，开箱即用，让你把时间留给模型创新而非重复造轮子。

---

## ✨ 核心特性

- **📦 31 个模块，105+ 算法** — 覆盖数模国赛 95% 以上的题型
- **🎯 统一 API** — 所有模型遵循 `创建 → fit() → 结果` 的一致接口
- **📊 内置可视化** — 每个模型自带绘图方法，论文级图表一键生成
- **📝 论文写作辅助** — 摘要自动生成、模型评价、LaTeX 报告、200+ 条语料库
- **🔬 12 道真题实战验证** — 经过 2019-2024 年国赛 A/B/C 三类题目的端到端测试
- **✅ 230 个单元测试** — 覆盖全部 31 个模块，边界检查完善
- **🖨️ 丰富的 `__repr__`** — 所有模型类均有友好的打印输出，方便调试

---

## 🚀 快速开始

### 安装

```bash
# 从 GitHub 安装
pip install git+https://github.com/HyperCharon/mathematical-modeling.git

# 或克隆后安装
git clone https://github.com/HyperCharon/mathematical-modeling.git
cd mathematical-modeling
pip install -e ".[dev]"

# 完整安装 (含深度学习)
pip install -e ".[full]"

# 启动 Web UI
pip install -e ".[streamlit]"
streamlit run app.py
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
print(f"CR = {ahp.CR:.4f} {'✅' if ahp.is_consistent else '❌'}")

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

---

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
| DEA | `DEA` | 数据包络分析，CCR/BCC 模型，效率评估 |
| PROMETHEE | `PROMETHEE` | 偏好排序方法，支持多种偏好函数 |

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
| 非线性规划 | `NonlinearProgramming` | 带约束非线性优化，SLSQP/COBYLA |
| 遗传算法 | `GeneticAlgorithm` | 通用 GA 框架，实数编码 |
| 粒子群 | `PSO` | 自适应惯性权重 PSO |
| 模拟退火 | `SimulatedAnnealing` | Metropolis 准则，高斯扰动 |
| 多目标优化 | `NSGA2` | Pareto 前沿求解，SBX 交叉 |
| 多起点优化 | `multi_start_optimize` | 多次随机初始化取最优 |

### 图论 (`mathflow.graph`)

| 模型 | 类名 | 说明 |
|------|------|------|
| 最短路径 | `ShortestPath` | Dijkstra / Bellman-Ford / Floyd |
| 最小生成树 | `MinSpanningTree` | Kruskal / Prim |
| 网络流 | `NetworkFlow` | 最大流 (Edmonds-Karp) + 最小割 |
| TSP | `TSPSolver` | 暴力/最近邻/2-opt/遗传算法 |
| 匈牙利算法 | `Hungarian` | 最小成本/最大收益指派 |
| 关键路径 | `CPM` | 前推/后推，关键路径识别，甘特图 |
| 网络分析 | `NetworkAnalysis` | 中心性指标、PageRank、社区发现 |

### 仿真 (`mathflow.simulation`)

| 模型 | 类名 | 说明 |
|------|------|------|
| 蒙特卡洛 | `MonteCarlo` | 积分、pi估算、风险分析 |
| 排队论 | `QueueModel` | M/M/1、M/M/c、M/M/1/K |
| 元胞自动机 | `CellularAutomata` | Game of Life、SIR、森林火灾 |

### 常微分方程 (`mathflow.ode`)

| 模型 | 类名 | 说明 |
|------|------|------|
| ODE 求解器 | `ODESolver` | Euler / Heun / RK4 / scipy，支持方程组 |
| 事件驱动 ODE | `EventODESolver` | 零点检测、终止事件、线性插值 |
| SIR 传染病 | `sir_model()` | 经典 SIR，可调 β/γ |
| Lotka-Volterra | `lotka_volterra()` | 捕食者-被捕食者动力学 |
| 阻尼振动 | `damped_oscillator()` | 二阶ODE转一阶方程组 |
| 一维热传导 | `heat_equation_1d()` | 有限差分法求解 PDE |
| 一维波动方程 | `wave_equation_1d()` | 波的传播与反射 |

### 统计分析 (`mathflow.stats`)

| 模型 | 类名 | 说明 |
|------|------|------|
| 多元回归 | `MultiRegression` | 完整统计报告 (R², F, t, p, DW)，回归诊断图 |
| 方差分析 | `ANOVA` | 单因素 ANOVA，自动显著性判断 |
| 相关性分析 | `CorrelationAnalysis` | Pearson/Spearman/Kendall 相关系数，显著性检验 |
| 灵敏度分析 | `SensitivityAnalysis` | OAT 单因素 / Morris 筛选 / Sobol 全局，龙卷风图 |

### 机器学习 (`mathflow.ml`)

| 模型 | 类名 | 说明 |
|------|------|------|
| 聚类分析 | `ClusterAnalysis` | KMeans/DBSCAN/层次聚类，自动选K，外部评估(ARI/NMI) |
| 分类器 | `Classifier` | Logistic/KNN/SVM/决策树/随机森林/XGBoost，多方法对比 |
| 降维分析 | `DimensionReduction` | PCA 主成分分析，双标图 |

### 插值与拟合 (`mathflow.interpolate`)

| 模型 | 类名 | 说明 |
|------|------|------|
| 曲线拟合 | `CurveFitter` | 自动拟合 (线性/多项式/指数/对数/幂/Logistic) |
| 插值器 | `Interpolator` | Lagrange / 线性 / 三次样条插值 |

### 博弈论 (`mathflow.game`)

| 模型 | 类名 | 说明 |
|------|------|------|
| Nash 均衡 | `NashEquilibrium` | 纯策略/混合策略/优势策略/最优响应 |
| 矩阵博弈 | `MatrixGame` | 零和博弈 (鞍点/线性规划求解) |

### 动态规划 (`mathflow.dp`)

| 模型 | 类名 | 说明 |
|------|------|------|
| 背包问题 | `Knapsack` | 0-1 背包、完全背包，DP 表可视化 |
| 资源分配 | `ResourceAllocation` | 将有限资源分配给多个活动 |

### 时间序列 (`mathflow.timeseries`)

| 模型 | 类名 | 说明 |
|------|------|------|
| 序列分解 | `TimeSeriesDecompose` | 加法/乘法模型，移动平均/STL 分解 |
| 平稳性检验 | `StationarityTest` | ADF/KPSS 检验，差分阶数建议 |

### 模糊逻辑 (`mathflow.fuzzy`)

| 模型 | 类名 | 说明 |
|------|------|------|
| 隶属函数 | `MembershipFunction` | 三角/梯形/高斯/S/Z/钟形 (7种) |
| 模糊推理 | `FuzzyInference` | Mamdani 型推理系统 |

### 马尔可夫链 (`mathflow.markov`)

| 模型 | 类名 | 说明 |
|------|------|------|
| 马尔可夫链 | `MarkovChain` | 稳态分布/n步转移/模拟/吸收概率/状态分类 |

### 概率统计 (`mathflow.prob`)

| 模型 | 类名 | 说明 |
|------|------|------|
| 分布拟合 | `DistributionFitter` | 8种分布自动拟合/AIC/BIC/K-S检验 |
| 假设检验 | `HypothesisTest` | t/卡方/正态性/方差齐性/Mann-Whitney |

### 灰色系统 (`mathflow.grey`)

| 模型 | 类名 | 说明 |
|------|------|------|
| GM(1,N) | `GM1N` | 灰色多变量预测 |
| 灰色决策 | `GreyDecision` | 灰色关联评价/聚类评价 |

### 小波分析 (`mathflow.wavelet`)

| 模型 | 类名 | 说明 |
|------|------|------|
| 小波变换 | `WaveletTransform` | DWT 分解重构/信号去噪/能量分布 |

### 📝 论文写作 (`mathflow.paper`)

| 模型 | 类名 | 说明 |
|------|------|------|
| 摘要生成 | `AbstractGenerator` | 五段式摘要 (背景→问题1→问题2→优势→关键词) |
| 模型评价 | `ModelEvaluator` | 自动生成优缺点/改进/推广，龙卷风图描述 |
| 章节生成 | `SectionGenerator` | 模型假设/符号说明(三线表)/问题分析/参考文献 |
| 一键论文 | `FullPaper` | 生成完整 LaTeX 论文框架，含全部标准章节 |
| 语料库 | `Phrases` | 200+ 条常用句式: 摘要/过渡词/模型描述/结果分析 |

### 辅助模块

| 模块 | 说明 |
|------|------|
| `mathflow.report` | LaTeX 报告自动生成 (公式/表格/图片，AHP/TOPSIS/回归结果自动转 LaTeX) |
| `mathflow.data` | 数据加载器 (CSV/Excel/JSON) + 数据清洗器 (缺失值/异常值/标准化) |
| `mathflow.viz` | 论文风格可视化主题 |

---

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
from mathflow.optimize import LinearProgramming, GeneticAlgorithm

# 线性规划
lp = LinearProgramming()
lp.set_objective([4, 3], sense="max")
lp.add_constraint([2, 1], "<=", 10)
lp.add_constraint([1, 1], "<=", 8)
result = lp.solve()
fig = lp.plot_feasible_region()  # 可行域可视化

# 遗传算法 (非线性优化)
ga = GeneticAlgorithm(fitness_func, n_vars=2, bounds=[(0, 10)]*2)
result = ga.run()
```

### ODE 动力学建模

```python
from mathflow.ode import ODESolver, sir_model

# SIR 传染病模型
solver = ODESolver(sir_model(beta=0.3, gamma=0.1), y0=[0.99, 0.01, 0])
result = solver.solve(t_span=(0, 160), dt=0.1)
fig = solver.plot(labels=["易感者", "感染者", "恢复者"])
```

### 论文写作

```python
from mathflow.paper import AbstractGenerator, FullPaper

# 自动生成摘要
ag = AbstractGenerator("某问题")
ag.add_problem_solution("问题描述", "AHP方法", "得分0.85", "评价模型")
ag.set_keywords(["AHP", "TOPSIS", "灵敏度分析"])
print(ag.generate())

# 一键生成完整论文
paper = FullPaper("2024年C题 农作物种植策略")
paper.add_sub_problem("最优种植方案", "线性规划", "收益285万")
paper.generate("paper.tex")
```

---

## 🔬 真题实战验证

经过 **12 道国赛真题** 的端到端测试：

| 年份 | 题目 | 类型 | 测试模块 |
|------|------|------|---------|
| 2024C | 农作物种植策略 | 数据优化 | 预测/优化/评价/概率 |
| 2024A | 板凳龙 | 运动学 | ODE/插值/模糊/马尔可夫 |
| 2024B | 生产决策 | 调度优化 | LP/IP/GA/综合压测 |
| 2023C | 蔬菜定价 | 预测型 | 时间序列/ARIMA/回归 |
| 2022C | 古代玻璃鉴别 | 数据分析 | ML分类/聚类/PCA |
| 2022A | 波浪能 | ODE | ODE/事件驱动/灵敏度 |
| 2022B | 无人机定位 | 估计 | 回归/蒙特卡洛/GM1N |
| 2022D | 气象报文 | 异常检测 | 平稳性/假设检验/小波 |
| 2021B | 乙醇偶合制备 | 统计型 | 回归/ANOVA/NSGA-II |
| 2020A | 炉温曲线 | PDE | ODE热传导/优化 |
| 2020B | 穿越沙漠 | 路径规划 | DP/图论/TSP/博弈论 |
| 2020C | 信贷决策 | 数据驱动 | 分类/聚类/AHP/TOPSIS |
| 2019A | 高压油管 | ODE控制 | ODE/GA/PSO/灵敏度 |
| 2019C | 机场出租车 | 排队论 | 排队论/蒙特卡洛/博弈 |

---

## 📁 项目结构

```
mathematical-modeling/
├── mathflow/
│   ├── evaluate/      # 评价类 (AHP, TOPSIS, 熵权法, CRITIC, GRA, 模糊, RSR)
│   ├── predict/       # 预测类 (灰色预测, ARIMA, 指数平滑, 回归)
│   ├── optimize/      # 优化类 (LP, IP, GA, PSO, SA, NSGA-II, 多起点)
│   ├── graph/         # 图论 (最短路, MST, 网络流, TSP, 匈牙利, CPM)
│   ├── simulation/    # 仿真 (蒙特卡洛, 排队论)
│   ├── ode/           # 常微分方程 (ODE求解器, 事件驱动, SIR, 热传导, 波动)
│   ├── stats/         # 统计 (多元回归, ANOVA, 灵敏度分析)
│   ├── ml/            # 机器学习 (聚类, 分类, PCA)
│   ├── interpolate/   # 插值与拟合
│   ├── game/          # 博弈论 (Nash均衡, 矩阵博弈)
│   ├── dp/            # 动态规划 (背包, 资源分配)
│   ├── timeseries/    # 时间序列 (分解, 平稳性检验)
│   ├── fuzzy/         # 模糊逻辑 (隶属函数, 模糊推理)
│   ├── markov/        # 马尔可夫链
│   ├── prob/          # 概率统计 (分布拟合, 假设检验)
│   ├── grey/          # 灰色系统 (GM(1,N), 灰色决策)
│   ├── wavelet/       # 小波分析
│   ├── paper/         # 论文写作 (摘要/评价/章节/语料库/一键论文)
│   ├── report/        # LaTeX 报告生成
│   ├── templates/     # 赛题模板
│   ├── data/          # 数据处理
│   ├── viz/           # 可视化
│   └── core/          # 核心工具
├── examples/          # 23 个完整示例
├── tests/             # 214 个单元测试
├── app.py             # Streamlit Web UI
└── pyproject.toml     # 项目配置
```

---

## 🧪 测试

```bash
# 运行全部测试
python -m pytest tests/ -v

# 运行特定模块测试
python -m pytest tests/test_evaluate.py -v
python -m pytest tests/test_ode.py -v

# 运行示例
python examples/01_ahp_topsis_demo.py
python examples/05_sir_epidemic.py
python examples/16_paper_writing.py
```

---

## 🤝 参考项目

本项目参考了以下优秀项目：

- [zhanwen/MathModel](https://github.com/zhanwen/MathModel) — 数模论文资源合集 (10k+ stars)
- [datawhalechina/intro-mathmodel](https://github.com/datawhalechina/intro-mathmodel) — Datawhale 数模教程
- [Valdecy/pyDecision](https://github.com/Valdecy/pyDecision) — 多准则决策库
- [HuangCongQing/Algorithms_MathModels](https://github.com/HuangCongQing/Algorithms_MathModels) — 算法与数学模型
- [jihe520/MathModelAgent](https://github.com/jihe520/MathModelAgent) — AI 数模 Agent

---

## 📄 License

MIT License

---

**如果觉得有用，给个 ⭐ 吧！**
