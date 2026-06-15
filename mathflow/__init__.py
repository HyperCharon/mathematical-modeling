"""
MathFlow — 数学建模竞赛全流程工具库
Mathematical Modeling Competition Toolkit

Modules:
    - evaluate:  评价类模型 (AHP, TOPSIS, 熵权法, CRITIC, 灰色关联, 模糊综合评价)
    - predict:   预测类模型 (灰色预测, ARIMA, 指数平滑, 回归预测)
    - optimize:  优化类模型 (线性规划, 整数规划, 遗传算法, 粒子群, 模拟退火)
    - ml:        机器学习 (聚类, 分类, 降维)
    - graph:     图论与网络 (最短路, 最小生成树, 网络流)
    - ode:       微分方程 (ODE求解, 参数拟合)
    - simulation:仿真模块 (蒙特卡洛, 排队论, 元胞自动机)
    - viz:       可视化引擎
    - data:      数据处理
    - report:    报告生成
"""

__version__ = "0.1.0"
__author__ = "Charon0415"

from mathflow.evaluate import ahp, topsis, entropy_weight, critic, gra, fuzzy_eval
from mathflow.predict import gray, arima_model, exponential_smooth
from mathflow.optimize import linear_prog, integer_prog, genetic_algo, pso
