"""
示例 15: 时间序列分解 + 平稳性检验

展示时间序列分析的完整流程。
"""

import sys
sys.path.insert(0, "..")
import numpy as np

from mathflow.timeseries import TimeSeriesDecompose, StationarityTest


def main():
    print("=" * 60)
    print("  示例: 时间序列分析")
    print("=" * 60)

    # 生成模拟数据: 趋势 + 季节性 + 噪声
    np.random.seed(42)
    t = np.arange(120)
    trend = 0.5 * t
    seasonal = 15 * np.sin(2 * np.pi * t / 12)
    noise = np.random.randn(120) * 3
    data = 50 + trend + seasonal + noise

    # ======== 1. 时间序列分解 ========
    print("\n【1】时间序列分解 (加法模型)")
    print("-" * 40)

    ts = TimeSeriesDecompose(data, period=12, model="additive")
    result = ts.decompose()
    print(ts.summary())

    fig = ts.plot()
    fig.savefig("15_decompose.png", dpi=150, bbox_inches="tight")
    print("  ✅ 已保存: 15_decompose.png")

    # ======== 2. 平稳性检验 ========
    print("\n\n【2】平稳性检验")
    print("-" * 40)

    # 原始序列 (非平稳)
    print("  原始序列:")
    st = StationarityTest(data)
    st_result = st.test_all()
    print(st.summary())

    # 差分后 (应平稳)
    print("\n  一阶差分后:")
    st2 = StationarityTest(np.diff(data))
    st2_result = st2.test_all()
    print(st2.summary())

    # ======== 3. 乘法模型分解 ========
    print("\n\n【3】乘法模型分解 (适合增长型数据)")
    print("-" * 40)

    # 生成乘法型数据
    data_mult = (1 + 0.01 * t) * (1 + 0.2 * np.sin(2 * np.pi * t / 12)) * (1 + np.random.randn(120) * 0.05)
    data_mult = np.abs(data_mult) * 100

    ts_mult = TimeSeriesDecompose(data_mult, period=12, model="multiplicative")
    result_mult = ts_mult.decompose()
    print(ts_mult.summary())


if __name__ == "__main__":
    main()
