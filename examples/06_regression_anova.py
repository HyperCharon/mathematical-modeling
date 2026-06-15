"""
示例 6: 多元回归 + 方差分析

场景: 类似 2021B 乙醇偶合制备C4烯烃 — 工艺参数优化
- 自变量: 温度、浓度、催化剂配比、反应时间
- 因变量: C4烯烃收率
"""

import sys
sys.path.insert(0, "..")
import numpy as np

from mathflow.stats import MultiRegression, ANOVA


def generate_chemical_data(n=60, seed=42):
    """模拟化工实验数据."""
    np.random.seed(seed)
    temp = np.random.uniform(200, 400, n)        # 温度 (°C)
    conc = np.random.uniform(0.1, 1.0, n)        # 浓度 (mol/L)
    ratio = np.random.uniform(1, 5, n)           # 催化剂配比
    time = np.random.uniform(0.5, 4.0, n)        # 反应时间 (h)

    # 真实关系: 收率 = f(温度, 浓度, 配比, 时间) + 噪声
    yield_ = (0.05 * temp + 15 * conc + 3 * ratio + 5 * time
              - 0.00008 * temp**2 - 5 * conc**2
              + 2 * conc * ratio
              + np.random.randn(n) * 2)

    X = np.column_stack([temp, conc, ratio, time])
    return X, yield_


def main():
    print("=" * 60)
    print("  示例: 多元回归 + 方差分析 — 化工工艺优化")
    print("=" * 60)

    # 生成数据
    X, y = generate_chemical_data(n=60)
    var_names = ["温度", "浓度", "配比", "时间"]

    # ======== 多元回归 ========
    print("\n【1】多元线性回归")
    print("-" * 40)

    model = MultiRegression(X, y, var_names=var_names)
    model.fit()
    print(model.summary())

    # 诊断图
    fig = model.plot_diagnostics()
    fig.savefig("06_regression_diagnostics.png", dpi=150, bbox_inches="tight")
    print("  ✅ 已保存: 06_regression_diagnostics.png")

    # ======== 方差分析 ========
    print("\n\n【2】方差分析 — 不同温度区间的收率差异")
    print("-" * 40)

    # 按温度分组
    temp = X[:, 0]
    y_low = y[temp < 280]
    y_mid = y[(temp >= 280) & (temp < 340)]
    y_high = y[temp >= 340]

    anova = ANOVA()
    anova.one_way(y_low, y_mid, y_high, group_names=["低温(<280°C)", "中温(280-340°C)", "高温(>340°C)"])
    print(anova.summary())

    # ======== 预测 ========
    print("\n\n【3】最优工艺参数预测")
    print("-" * 40)

    # 网格搜索最优参数
    best_yield = -float("inf")
    best_params = None
    for temp in np.arange(250, 400, 10):
        for conc in np.arange(0.3, 1.0, 0.1):
            for ratio in np.arange(2, 5, 0.5):
                for time in np.arange(1, 4, 0.5):
                    pred = model.predict([[temp, conc, ratio, time]])[0]
                    if pred > best_yield:
                        best_yield = pred
                        best_params = [temp, conc, ratio, time]

    print(f"  最优参数:")
    for name, val in zip(var_names, best_params):
        print(f"    {name} = {val}")
    print(f"  预测收率 = {best_yield:.2f}")


if __name__ == "__main__":
    main()
