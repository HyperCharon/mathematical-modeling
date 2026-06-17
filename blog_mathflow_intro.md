# MathFlow：我做了一个覆盖数学建模全流程的 Python 工具库

> 做数学建模竞赛时，你有没有这样的经历：每次比赛都要重新写 AHP、TOPSIS、灰色预测的代码，调试半天才发现权重归一化写错了？我做了 MathFlow，一个开箱即用的数学建模工具库，31 个模块、105+ 算法、230 个测试，让你把时间留给模型创新而非重复造轮子。

## 为什么做 MathFlow

参加过数学建模的同学都知道，比赛时间紧、任务重，三天时间要完成选题、建模、求解、写作全流程。但很多时间其实花在了"造轮子"上：

- **评价类问题**：每次都要手写 AHP 判断矩阵、一致性检验、TOPSIS 归一化……
- **预测类问题**：灰色预测的 GM(1,1) 公式背不下来，ARIMA 的参数选择靠蒙……
- **优化类问题**：线性规划的约束条件写错一个符号，整个模型就废了……
- **论文写作**：摘要怎么写？模型评价怎么组织？LaTeX 排版又出问题了……

这些问题本质上都是**重复劳动**。如果有一个工具库，把这些常用算法都封装好，统一 API，开箱即用，那该多好？

于是 MathFlow 诞生了。

## MathFlow 是什么

MathFlow 是一个一站式 Python 工具库，覆盖数学建模竞赛中的全链路：

```
评价 → 预测 → 优化 → 图论 → 仿真 → ODE → 统计 → ML → 博弈论 → 动态规划 → 时间序列 → 模糊逻辑 → 马尔可夫 → 灰色系统 → 小波分析 → 论文写作
```

**核心数据：**
- 📦 31 个模块，105+ 算法
- 🎯 统一 API 设计
- 📊 内置可视化
- 📝 论文写作辅助
- 🔬 12 道国赛真题验证
- ✅ 230 个单元测试

## 快速上手

### 安装

```bash
pip install git+https://github.com/HyperCharon/mathematical-modeling.git
```

### 5 分钟示例

```python
from mathflow.evaluate import AHP, TOPSIS, EntropyWeight
import numpy as np

# AHP 层次分析法
ahp = AHP()
ahp.set_matrix([
    [1,   3,   5],
    [1/3, 1,   3],
    [1/5, 1/3, 1]
])
ahp.fit()
print(f"权重: {ahp.weights}")
print(f"CR = {ahp.CR:.4f} {'✅' if ahp.is_consistent else '❌'}")

# TOPSIS 综合评价
data = np.array([
    [80, 90, 85, 70],   # 方案1
    [70, 80, 90, 80],   # 方案2
    [90, 85, 80, 75],   # 方案3
])
topsis = TOPSIS(data, weights=[0.3, 0.25, 0.25, 0.2], types=[1, 1, 1, -1])
topsis.fit()
print(f"排名: {topsis.rankings}")
```

就这么简单。不需要理解底层公式，不需要调试边界条件，三行代码搞定一个完整的评价模型。

## 设计哲学

### 1. 统一 API

所有模型遵循一致的接口设计：

| 模块类型 | API 模式 |
|---------|---------|
| 评价/预测/统计/ML | `fit()` → `result` |
| 图论/规划 | `solve()` → `result` |
| 优化算法 (GA/PSO/SA) | `run()` → `result` |
| 仿真 | `run()` 或 `solve()` |

你只需要记住一个模式：**创建对象 → 调用方法 → 获取结果**。

### 2. 内置可视化

每个模型自带绘图方法，论文级图表一键生成：

```python
# AHP 权重柱状图
fig = ahp.plot_weights()

# TOPSIS 排名雷达图
fig = toplot_radar()

# 灰色预测拟合图
fig = gp.plot()

# 灵敏度分析龙卷风图
fig = sa.plot_tornado()
```

### 3. 完整的错误处理

每个模型都有完善的输入验证和错误提示：

```python
ahp = AHP()
ahp.fit()  # RuntimeError: 请先调用 set_matrix() 设置判断矩阵

gp = GreyPrediction([1, 2])  # ValueError: 灰色预测至少需要 4 个数据点
```

不会给你一个莫名其妙的 `NoneType has no attribute`。

### 4. 友好的打印输出

所有模型都有 `__repr__` 方法，调试时一目了然：

```python
>>> ahp
AHP(method='eigenvalue')

>>> ahp.fit()
AHP(n=3, method='eigenvalue', CR=0.0332)

>>> topsis
TOPSIS(n_samples=3, n_indicators=4)
```

## 覆盖的算法

### 评价类 (每次比赛必用)

| 模型 | 说明 |
|------|------|
| AHP | 层次分析法，支持特征值法/几何平均法/算术平均法 |
| TOPSIS | 逼近理想解排序 |
| 熵权法 | 基于信息熵的客观赋权 |
| CRITIC | 基于对比强度和冲突性的客观赋权 |
| 灰色关联 | 序列几何相似度分析 |
| 模糊综合评价 | 定性评价定量化 |
| 秩和比法 (RSR) | 融合参数与非参数统计 |
| DEA | 数据包络分析，CCR/BCC 模型 |
| PROMETHEE | 偏好排序方法 |

### 预测类

| 模型 | 说明 |
|------|------|
| 灰色预测 | GM(1,1)，小样本指数增长预测 |
| ARIMA | 自回归积分移动平均，自动选参 |
| 指数平滑 | 一次/Holt/Holt-Winters |
| 回归预测 | 线性/多项式/岭/Lasso 回归 |

### 优化类

| 模型 | 说明 |
|------|------|
| 线性规划 | 可视化可行域 |
| 整数规划 | 纯整数/混合整数/0-1 规划 |
| 非线性规划 | SLSQP/COBYLA |
| 遗传算法 | 通用 GA 框架 |
| 粒子群 (PSO) | 自适应惯性权重 |
| 模拟退火 | Metropolis 准则 |
| NSGA-II | 多目标优化，Pareto 前沿 |

### 图论

| 模型 | 说明 |
|------|------|
| 最短路径 | Dijkstra / Bellman-Ford / Floyd |
| 最小生成树 | Kruskal / Prim |
| 网络流 | 最大流 (Edmonds-Karp) + 最小割 |
| TSP | 暴力/最近邻/2-opt/遗传算法 |
| 匈牙利算法 | 最小成本指派 |
| 关键路径 (CPM) | 前推/后推，甘特图 |
| 网络分析 | 中心性指标、PageRank、社区发现 |

### 更多

还有 ODE 求解器、蒙特卡洛仿真、排队论、元胞自动机、聚类分析、分类器、PCA、博弈论、动态规划、时间序列分解、模糊逻辑、马尔可夫链、概率分布拟合、假设检验、灰色系统、小波分析、论文写作……

完整列表请看 [README](https://github.com/HyperCharon/mathematical-modeling)。

## 实战验证

MathFlow 经过 12 道国赛真题的端到端测试：

| 年份 | 题目 | 类型 | 测试模块 |
|------|------|------|---------|
| 2024C | 农作物种植策略 | 数据优化 | 预测/优化/评价/概率 |
| 2024A | 板凳龙 | 运动学 | ODE/插值/模糊/马尔可夫 |
| 2024B | 生产决策 | 调度优化 | LP/IP/GA/综合压测 |
| 2023C | 蔬菜定价 | 预测型 | 时间序列/ARIMA/回归 |
| 2022C | 古代玻璃鉴别 | 数据分析 | ML分类/聚类/PCA |
| 2022A | 波浪能 | ODE | ODE/事件驱动/灵敏度 |
| 2020B | 穿越沙漠 | 路径规划 | DP/图论/TSP/博弈论 |
| 2019A | 高压油管 | ODE控制 | ODE/GA/PSO/灵敏度 |

每道真题都从数据处理到结果输出完整跑通，确保在真实比赛场景下可靠。

## 代码质量

MathFlow 不只是算法的堆砌，更注重工程质量：

- **230 个单元测试** — 覆盖全部 31 个模块
- **完善的错误处理** — 每个模型都有 `_ensure_result()` 守卫
- **统一的 `__repr__`** — 所有 39 个类都有友好的打印输出
- **全局配置系统** — 通过 `Config` 统一管理阈值、样式、精度
- **未使用导入零容忍** — 37 个文件的导入都经过清理

## 论文写作辅助

MathFlow 不只是计算工具，还是论文写作助手：

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

内置 200+ 条常用句式：摘要、过渡词、模型描述、结果分析……

## 未来计划

MathFlow 还在持续进化中，接下来计划：

- **LSTM 时间序列预测** — 深度学习预测模型
- **VIKOR/ELECTRE** — 更多多准则决策方法
- **2D PDE 求解器** — 有限差分/有限元
- **Sphinx API 文档** — 完整的 API 参考
- **更多真题验证** — 覆盖到 2018 年

## 如何使用

```bash
# 克隆项目
git clone https://github.com/HyperCharon/mathematical-modeling.git
cd mathematical-modeling

# 安装
pip install -e ".[dev]"

# 运行示例
python examples/01_ahp_topsis_demo.py

# 运行测试
python -m pytest tests/ -v
```

## 写在最后

做这个项目的初衷很简单：**让数学建模的同学少写重复代码，多花时间在模型创新上**。

如果你觉得有用，给个 ⭐ 吧！

项目地址：[github.com/HyperCharon/mathematical-modeling](https://github.com/HyperCharon/mathematical-modeling)

---

*MathFlow — 让数学建模更高效*
