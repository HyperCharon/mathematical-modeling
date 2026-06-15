"""
示例 10: 曲线拟合与插值

展示自动曲线拟合和多种插值方法。
"""

import sys
sys.path.insert(0, "..")
import numpy as np

from mathflow.interpolate import CurveFitter, Interpolator


def main():
    print("=" * 60)
    print("  示例: 曲线拟合与插值")
    print("=" * 60)

    # ======== 1. 自动曲线拟合 ========
    print("\n【1】自动曲线拟合")
    print("-" * 40)

    # 指数增长数据
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.array([2.5, 4.8, 9.2, 18.1, 35.2, 68.5, 133.0, 258.1, 500.3, 970.1])

    cf = CurveFitter(x, y)
    cf.auto_fit()
    print(cf.summary())

    fig = cf.plot()
    fig.savefig("10_curve_fit.png", dpi=150, bbox_inches="tight")
    print("  ✅ 已保存: 10_curve_fit.png")

    # 预测
    x_new = [11, 12, 13]
    y_pred = cf.predict(x_new)
    print(f"\n  预测: x={x_new} → y={[f'{v:.1f}' for v in y_pred]}")

    # ======== 2. 多种拟合方法对比 ========
    print("\n\n【2】多种拟合方法对比")
    print("-" * 40)

    for method in ["linear", "polynomial2", "exponential", "power"]:
        try:
            cf2 = CurveFitter(x, y)
            cf2.fit(method)
            print(f"  {method:15s}: R²={cf2.result.r2:.6f}, 方程={cf2.result.equation}")
        except Exception as e:
            print(f"  {method:15s}: 失败 - {e}")

    # ======== 3. 插值方法 ========
    print("\n\n【3】插值方法对比")
    print("-" * 40)

    # 已知数据点 (正弦函数采样)
    x_known = np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    y_known = np.sin(x_known)

    interp = Interpolator(x_known, y_known)

    # 在 x=1.25 处插值
    x_test = 1.25
    y_true = np.sin(x_test)
    print(f"  真实值 sin({x_test}) = {y_true:.6f}")

    for method in ["lagrange", "linear", "cubic_spline"]:
        y_interp = interp.interpolate(x_test, method=method)
        error = abs(y_interp - y_true)
        print(f"  {method:15s}: {y_interp:.6f}, 误差={error:.2e}")

    fig2 = interp.plot()
    fig2.savefig("10_interpolation.png", dpi=150, bbox_inches="tight")
    print("  ✅ 已保存: 10_interpolation.png")


if __name__ == "__main__":
    main()
