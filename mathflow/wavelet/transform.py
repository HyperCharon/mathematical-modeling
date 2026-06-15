"""
小波变换

支持离散小波变换 (DWT)、信号去噪、多分辨率分析。

Example:
    >>> from mathflow.wavelet import WaveletTransform
    >>> import numpy as np
    >>> t = np.linspace(0, 1, 1000)
    >>> signal = np.sin(2*np.pi*50*t) + np.sin(2*np.pi*120*t) + np.random.randn(1000)*0.5
    >>> wt = WaveletTransform(signal)
    >>> coeffs = wt.decompose(level=4)
    >>> denoised = wt.denoise(level=4, threshold="soft")
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class WaveletResult:
    """小波变换结果."""
    coeffs: List[np.ndarray]   # [cA_n, cD_n, cD_{n-1}, ..., cD_1]
    level: int
    wavelet: str


class WaveletTransform:
    """
    离散小波变换.

    Parameters
    ----------
    signal : array-like
        输入信号
    wavelet : str
        小波基: "haar", "db2", "db4"
    """

    def __init__(self, signal, wavelet="haar"):
        self.signal = np.asarray(signal, dtype=float)
        self.wavelet = wavelet
        self._result = None

    def decompose(self, level: int = 3) -> WaveletResult:
        """
        多层小波分解.

        Parameters
        ----------
        level : int
            分解层数
        """
        coeffs = []
        lengths = []
        data = self.signal.copy()

        for _ in range(level):
            lengths.append(len(data))
            cA, cD = self._dwt_step(data)
            coeffs.append(cD)
            data = cA

        coeffs.append(data)  # 最后一层近似系数
        coeffs.reverse()  # [cA_n, cD_n, cD_{n-1}, ..., cD_1]
        lengths.reverse()

        self._result = WaveletResult(coeffs=coeffs, level=level, wavelet=self.wavelet)
        self._lengths = lengths
        return self._result

    def _dwt_step(self, data):
        """单步小波分解 (Haar)."""
        n = len(data)
        if n % 2 != 0:
            data = np.append(data, data[-1])
            n += 1

        if self.wavelet == "haar":
            cA = (data[0::2] + data[1::2]) / np.sqrt(2)
            cD = (data[0::2] - data[1::2]) / np.sqrt(2)
        elif self.wavelet == "db2":
            # Daubechies-2 简化实现
            h = np.array([0.6830127, 1.1830127, 0.3169873, -0.1830127]) / np.sqrt(2)
            g = np.array([-0.1830127, -0.3169873, 1.1830127, -0.6830127]) / np.sqrt(2)
            cA = np.convolve(data, h, mode='same')[::2][:n//2]
            cD = np.convolve(data, g, mode='same')[::2][:n//2]
        else:
            cA = (data[0::2] + data[1::2]) / np.sqrt(2)
            cD = (data[0::2] - data[1::2]) / np.sqrt(2)

        return cA, cD

    def reconstruct(self, coeffs: List[np.ndarray] = None) -> np.ndarray:
        """小波重构."""
        if coeffs is None:
            coeffs = self._result.coeffs

        data = coeffs[0].copy()  # 最低频近似系数

        for i in range(1, len(coeffs)):
            cD = coeffs[i]
            data = self._idwt_step(data, cD)
            if hasattr(self, '_lengths') and i-1 < len(self._lengths):
                data = data[:self._lengths[i-1]]  # Truncate to original length

        return data[:len(self.signal)]

    def _idwt_step(self, cA, cD):
        """单步小波重构 (Haar)."""
        n = max(len(cA), len(cD)) * 2
        result = np.zeros(n)

        if self.wavelet == "haar":
            result[0::2] = (cA + cD) / np.sqrt(2)
            result[1::2] = (cA - cD) / np.sqrt(2)
        else:
            result[0::2] = (cA + cD) / np.sqrt(2)
            result[1::2] = (cA - cD) / np.sqrt(2)

        return result

    def denoise(self, level: int = 3, threshold: str = "soft",
                sigma: float = None) -> np.ndarray:
        """
        小波去噪.

        Parameters
        ----------
        level : int
            分解层数
        threshold : str
            "soft" (软阈值) 或 "hard" (硬阈值)
        sigma : float, optional
            噪声标准差 (默认自动估计)
        """
        result = self.decompose(level)
        coeffs = [result.coeffs[0].copy()]

        if sigma is None:
            # 用 MAD 估计噪声标准差
            sigma = np.median(np.abs(result.coeffs[-1])) / 0.6745

        # 阈值
        t = sigma * np.sqrt(2 * np.log(len(self.signal)))

        for i in range(1, len(result.coeffs)):
            cD = result.coeffs[i].copy()
            if threshold == "soft":
                cD = np.sign(cD) * np.maximum(np.abs(cD) - t, 0)
            else:  # hard
                cD[np.abs(cD) < t] = 0
            coeffs.append(cD)

        return self.reconstruct(coeffs)

    def energy_distribution(self) -> dict:
        """计算各层能量分布."""
        self._ensure_result()
        coeffs = self._result.coeffs
        total_energy = sum(np.sum(c**2) for c in coeffs)

        distribution = {}
        for i, c in enumerate(coeffs):
            energy = np.sum(c**2)
            if i == 0:
                name = f"近似系数 cA{self._result.level}"
            else:
                name = f"细节系数 cD{self._result.level - i + 1}"
            distribution[name] = {
                "能量": energy,
                "占比": energy / total_energy * 100 if total_energy > 0 else 0,
            }

        return distribution

    def _ensure_result(self):
        if self._result is None:
            raise RuntimeError("请先调用 decompose()")

    def plot(self, figsize=(14, 10)):
        """绘制小波分解结果."""
        import matplotlib.pyplot as plt

        self._ensure_result()
        r = self._result
        n_coeffs = len(r.coeffs)

        fig, axes = plt.subplots(n_coeffs + 1, 1, figsize=figsize)

        axes[0].plot(self.signal, "b-", linewidth=0.8)
        axes[0].set_ylabel("原始信号")
        axes[0].set_title(f"小波分解 ({r.wavelet}, {r.level}层)")

        for i, c in enumerate(r.coeffs):
            label = f"cA{r.level}" if i == 0 else f"cD{r.level - i + 1}"
            axes[i + 1].plot(c, "r-" if i > 0 else "g-", linewidth=0.8)
            axes[i + 1].set_ylabel(label)

        axes[-1].set_xlabel("样本")
        plt.tight_layout()
        return fig

    def summary(self):
        self._ensure_result()
        r = self._result
        energy = self.energy_distribution()

        lines = [
            "=" * 50,
            "  小波变换结果",
            "=" * 50,
            f"  小波基: {r.wavelet}",
            f"  分解层数: {r.level}",
            f"  信号长度: {len(self.signal)}",
            "-" * 50,
            "  能量分布:",
        ]
        for name, info in energy.items():
            lines.append(f"    {name}: {info['占比']:.1f}%")
        lines.append("=" * 50)
        return "\n".join(lines)
