"""
示例 17: 模糊逻辑 + 马尔可夫链 + 小波变换

展示三个新模块的实际应用。
"""

import sys
sys.path.insert(0, "..")
import numpy as np

from mathflow.fuzzy import MembershipFunction, FuzzyInference
from mathflow.markov import MarkovChain
from mathflow.wavelet import WaveletTransform
from mathflow.prob import DistributionFitter, HypothesisTest


def main():
    print("=" * 60)
    print("  示例: 模糊逻辑 + 马尔可夫链 + 小波变换 + 概率统计")
    print("=" * 60)

    # ======== 1. 模糊推理: 温度控制 ========
    print("\n【1】模糊推理 — 空调温控系统")
    print("-" * 40)

    fi = FuzzyInference()
    fi.add_input("温度", (0, 40), [
        ("冷", "gaussian", 10, 5),
        ("适中", "gaussian", 20, 5),
        ("热", "gaussian", 30, 5),
    ])
    fi.add_output("功率", (0, 100), [
        ("低", "triangle", 0, 0, 50),
        ("中", "triangle", 25, 50, 75),
        ("高", "triangle", 50, 100, 100),
    ])
    fi.add_rule({"温度": "冷"}, "功率", "高")
    fi.add_rule({"温度": "适中"}, "功率", "低")
    fi.add_rule({"温度": "热"}, "功率", "高")

    for temp in [5, 15, 20, 25, 35]:
        result = fi.infer({"温度": temp})
        print(f"  温度={temp}°C → 功率={result['功率']:.1f}%")

    # ======== 2. 马尔可夫链: 天气预测 ========
    print("\n\n【2】马尔可夫链 — 天气状态预测")
    print("-" * 40)

    P = np.array([
        [0.7, 0.2, 0.1],  # 晴→晴, 晴→阴, 晴→雨
        [0.3, 0.4, 0.3],  # 阴→晴, 阴→阴, 阴→雨
        [0.2, 0.3, 0.5],  # 雨→晴, 雨→阴, 雨→雨
    ])
    mc = MarkovChain(P, states=["晴天", "阴天", "雨天"])

    pi = mc.steady_state()
    print("  稳态分布:")
    for i, p in enumerate(pi):
        print(f"    {mc.states[i]}: {p:.4f}")

    print(f"\n  10步后转移概率:")
    P10 = mc.n_step_transition(10)
    for i in range(3):
        print(f"    从{mc.states[i]}出发: {dict(zip(mc.states, [f'{p:.3f}' for p in P10[i]]))}")

    # ======== 3. 小波变换: 信号去噪 ========
    print("\n\n【3】小波变换 — 信号去噪")
    print("-" * 40)

    np.random.seed(42)
    t = np.linspace(0, 1, 512)
    clean_signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 30 * t)
    noisy_signal = clean_signal + np.random.randn(512) * 0.5

    wt = WaveletTransform(noisy_signal)
    result = wt.decompose(level=4)
    print(wt.summary())

    denoised = wt.denoise(level=4)
    noise_rmse = np.sqrt(np.mean((noisy_signal - clean_signal)**2))
    denoise_rmse = np.sqrt(np.mean((denoised - clean_signal)**2))
    print(f"\n  去噪前 RMSE: {noise_rmse:.4f}")
    print(f"  去噪后 RMSE: {denoise_rmse:.4f}")
    print(f"  信噪比提升: {20*np.log10(noise_rmse/denoise_rmse):.1f} dB")

    # ======== 4. 概率分布拟合 ========
    print("\n\n【4】概率分布拟合")
    print("-" * 40)

    np.random.seed(42)
    data = np.concatenate([
        np.random.normal(50, 10, 150),
        np.random.normal(80, 5, 50),
    ])

    df = DistributionFitter(data)
    results = df.auto_fit()
    print(df.summary())

    # ======== 5. 假设检验 ========
    print("\n\n【5】假设检验")
    print("-" * 40)

    np.random.seed(42)
    group_a = np.random.normal(75, 10, 30)
    group_b = np.random.normal(82, 10, 30)

    ht = HypothesisTest()

    result1 = ht.two_sample_ttest(group_a, group_b)
    print(ht.summary(result1))

    result2 = ht.normality_test(group_a)
    print(ht.summary(result2))


if __name__ == "__main__":
    main()
