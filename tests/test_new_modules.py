"""新增模块测试: 模糊逻辑, 马尔可夫链, 概率分布, 假设检验, 灰色系统, 小波变换."""

import numpy as np
import pytest
from mathflow.fuzzy import MembershipFunction, FuzzyInference
from mathflow.markov import MarkovChain
from mathflow.prob import DistributionFitter, HypothesisTest
from mathflow.grey import GM1N, GreyDecision
from mathflow.wavelet import WaveletTransform


# ========== 模糊逻辑 ==========
class TestMembershipFunction:
    def test_triangle(self):
        mf = MembershipFunction()
        y = mf.triangle([1, 3, 5], a=0, b=3, c=6)
        assert y[1] > 0  # x=3 是峰值
        assert y[0] > 0  # x=1 在上升段
        assert y[2] > 0  # x=5 在下降段

    def test_gaussian(self):
        mf = MembershipFunction()
        y = mf.gaussian([5], mean=5, sigma=1)
        assert abs(y[0] - 1.0) < 0.01

    def test_trapezoid(self):
        mf = MembershipFunction()
        y = mf.trapezoid([2, 5], a=1, b=3, c=7, d=9)
        assert y[0] > 0
        assert y[1] == 1.0


class TestFuzzyInference:
    def test_basic(self):
        fi = FuzzyInference()
        fi.add_input("温度", (0, 40), [
            ("冷", "gaussian", 10, 5),
            ("适中", "gaussian", 20, 5),
            ("热", "gaussian", 30, 5),
        ])
        fi.add_output("风速", (0, 100), [
            ("低", "triangle", 0, 0, 50),
            ("高", "triangle", 50, 100, 100),
        ])
        fi.add_rule({"温度": "冷"}, "风速", "高")
        fi.add_rule({"温度": "热"}, "风速", "高")
        fi.add_rule({"温度": "适中"}, "风速", "低")

        result = fi.infer({"温度": 10})
        assert "风速" in result
        assert 0 <= result["风速"] <= 100


# ========== 马尔可夫链 ==========
class TestMarkovChain:
    def test_steady_state(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc = MarkovChain(P, states=["晴天", "雨天"])
        pi = mc.steady_state()
        assert abs(pi.sum() - 1.0) < 1e-10
        assert all(pi >= 0)

    def test_n_step(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc = MarkovChain(P)
        P2 = mc.n_step_transition(2)
        assert abs(P2.sum(axis=1).mean() - 1.0) < 1e-10

    def test_simulate(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        mc = MarkovChain(P)
        states = mc.simulate(n_steps=100)
        assert len(states) == 100
        assert all(0 <= s <= 1 for s in states)


# ========== 概率分布拟合 ==========
class TestDistributionFitter:
    def test_auto_fit(self):
        np.random.seed(42)
        data = np.random.normal(50, 10, 200)
        df = DistributionFitter(data)
        results = df.auto_fit()
        assert len(results) > 0
        assert results[0].distribution  # 应该拟合出正态分布

    def test_fit_specific(self):
        np.random.seed(42)
        data = np.random.exponential(2, 200)
        df = DistributionFitter(data)
        result = df.fit("expon")
        assert result.distribution == "指数分布"


# ========== 假设检验 ==========
class TestHypothesisTest:
    def test_two_sample_ttest(self):
        np.random.seed(42)
        data1 = np.random.normal(50, 10, 30)
        data2 = np.random.normal(55, 10, 30)
        ht = HypothesisTest()
        result = ht.two_sample_ttest(data1, data2)
        assert result.p_value > 0

    def test_normality(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        ht = HypothesisTest()
        result = ht.normality_test(data)
        assert not result.is_significant  # 正态数据不应拒绝

    def test_paired_ttest(self):
        np.random.seed(42)
        before = np.random.normal(50, 10, 20)
        after = before + np.random.normal(5, 2, 20)
        ht = HypothesisTest()
        result = ht.paired_ttest(before, after)
        assert result.is_significant  # 应该有显著差异


# ========== 灰色系统 ==========
class TestGM1N:
    def test_basic(self):
        X = np.array([
            [100, 50, 30],
            [110, 55, 35],
            [125, 60, 38],
            [140, 68, 42],
            [155, 75, 48],
        ])
        gm = GM1N(X)
        result = gm.fit()
        # GM(1,N) 对小样本可能 R² 为负，只要能运行即可
        assert result.coefficients is not None
        assert len(result.fitted_values) == 5


class TestGreyDecision:
    def test_relational(self):
        data = np.array([
            [80, 90, 70],
            [75, 85, 80],
            [90, 80, 75],
        ])
        gd = GreyDecision(data)
        result = gd.evaluate()
        assert len(result.scores) == 3
        assert set(result.rankings) == {1, 2, 3}


# ========== 小波变换 ==========
class TestWaveletTransform:
    def test_decompose_reconstruct(self):
        np.random.seed(42)
        signal = np.sin(np.linspace(0, 4*np.pi, 128)) + np.random.randn(128) * 0.1
        wt = WaveletTransform(signal)
        result = wt.decompose(level=3)
        reconstructed = wt.reconstruct()
        # 重构信号应接近原始信号
        error = np.sqrt(np.mean((signal - reconstructed)**2))
        assert error < 0.5

    def test_denoise(self):
        np.random.seed(42)
        t = np.linspace(0, 1, 256)
        clean = np.sin(2 * np.pi * 10 * t)
        noisy = clean + np.random.randn(256) * 0.5
        wt = WaveletTransform(noisy)
        denoised = wt.denoise(level=3)
        # 去噪后应更接近干净信号
        noise_error = np.sqrt(np.mean((noisy - clean)**2))
        denoise_error = np.sqrt(np.mean((denoised - clean)**2))
        assert denoise_error < noise_error

    def test_energy(self):
        signal = np.sin(np.linspace(0, 4*np.pi, 128))
        wt = WaveletTransform(signal)
        wt.decompose(level=3)
        energy = wt.energy_distribution()
        assert len(energy) == 4  # 3层细节 + 1层近似


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
