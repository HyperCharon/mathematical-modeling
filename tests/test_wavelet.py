"""
测试 wavelet 模块: WaveletTransform
"""

import numpy as np
import pytest


class TestWaveletTransform:
    """测试小波变换."""

    def test_decompose_coefficients_length(self):
        """分解后系数长度应为 level+1."""
        from mathflow.wavelet.transform import WaveletTransform
        signal = np.random.randn(64)
        wt = WaveletTransform(signal, wavelet="haar")
        result = wt.decompose(level=3)
        assert len(result.coeffs) == 4  # 1 近似 + 3 细节

    def test_decompose_before_reconstruct_raises(self):
        """decompose 前调用 reconstruct 应抛出异常."""
        from mathflow.wavelet.transform import WaveletTransform
        signal = np.random.randn(64)
        wt = WaveletTransform(signal, wavelet="haar")
        # 未 decompose 时 reconstruct 应抛出 AttributeError (NoneType)
        with pytest.raises(AttributeError):
            wt.reconstruct()

    def test_haar_constant_signal(self):
        """常数信号的细节系数应约为 0."""
        from mathflow.wavelet.transform import WaveletTransform
        signal = np.ones(64) * 5.0
        wt = WaveletTransform(signal, wavelet="haar")
        result = wt.decompose(level=3)
        # 细节系数应接近 0
        for detail in result.coeffs[1:]:
            assert np.all(np.abs(detail) < 1e-10)

    def test_reconstruct_preserves_length(self):
        """重构后信号长度应与原始信号相同."""
        from mathflow.wavelet.transform import WaveletTransform
        signal = np.random.randn(64)
        wt = WaveletTransform(signal, wavelet="haar")
        result = wt.decompose(level=3)
        reconstructed = wt.reconstruct(result.coeffs)
        assert len(reconstructed) == len(signal)

    def test_haar_reconstruction_accuracy(self):
        """Haar 小波重构应近似还原原始信号."""
        from mathflow.wavelet.transform import WaveletTransform
        signal = np.random.randn(64)
        wt = WaveletTransform(signal, wavelet="haar")
        result = wt.decompose(level=3)
        reconstructed = wt.reconstruct(result.coeffs)
        np.testing.assert_array_almost_equal(reconstructed, signal, decimal=10)

    def test_energy_distribution(self):
        """能量分布应返回字典且总和约 100%."""
        from mathflow.wavelet.transform import WaveletTransform
        signal = np.random.randn(64)
        wt = WaveletTransform(signal, wavelet="haar")
        wt.decompose(level=3)
        energy = wt.energy_distribution()
        assert isinstance(energy, dict)
        # 键应包含近似和细节系数
        assert any("近似" in k for k in energy.keys())
        assert any("细节" in k for k in energy.keys())
        # 每个值是 {"能量": ..., "占比": ...}
        total_percentage = sum(v["占比"] for v in energy.values())
        assert abs(total_percentage - 100) < 1

    def test_denoise_soft(self):
        """软阈值去噪应降低噪声."""
        from mathflow.wavelet.transform import WaveletTransform
        np.random.seed(42)
        t = np.linspace(0, 1, 128)
        clean = np.sin(2 * np.pi * 5 * t)
        noisy = clean + np.random.randn(128) * 0.5

        wt = WaveletTransform(noisy, wavelet="haar")
        denoised = wt.denoise(level=3, threshold="soft")
        # 去噪后 MSE 应更小
        mse_noisy = np.mean((noisy - clean)**2)
        mse_denoised = np.mean((denoised - clean)**2)
        assert mse_denoised < mse_noisy

    def test_denoise_hard(self):
        """硬阈值去噪应降低噪声."""
        from mathflow.wavelet.transform import WaveletTransform
        np.random.seed(42)
        t = np.linspace(0, 1, 128)
        clean = np.sin(2 * np.pi * 5 * t)
        noisy = clean + np.random.randn(128) * 0.5

        wt = WaveletTransform(noisy, wavelet="haar")
        denoised = wt.denoise(level=3, threshold="hard")
        mse_noisy = np.mean((noisy - clean)**2)
        mse_denoised = np.mean((denoised - clean)**2)
        assert mse_denoised < mse_noisy

    def test_summary(self):
        """summary 应返回字符串."""
        from mathflow.wavelet.transform import WaveletTransform
        signal = np.random.randn(64)
        wt = WaveletTransform(signal, wavelet="haar")
        wt.decompose(level=3)
        s = wt.summary()
        assert isinstance(s, str)
        assert "haar" in s.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
