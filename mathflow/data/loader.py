"""
数据加载器

支持多种格式的数据加载。

Example:
    >>> from mathflow.data import DataLoader
    >>> df = DataLoader.load_csv("data.csv")
    >>> df = DataLoader.load_excel("data.xlsx")
"""

import numpy as np
import pandas as pd


class DataLoader:
    """数据加载器."""

    @staticmethod
    def load_csv(filepath, encoding="utf-8", **kwargs) -> pd.DataFrame:
        """加载 CSV 文件."""
        return pd.read_csv(filepath, encoding=encoding, **kwargs)

    @staticmethod
    def load_excel(filepath, sheet_name=0, **kwargs) -> pd.DataFrame:
        """加载 Excel 文件."""
        return pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)

    @staticmethod
    def load_json(filepath, **kwargs) -> pd.DataFrame:
        """加载 JSON 文件."""
        return pd.read_json(filepath, **kwargs)

    @staticmethod
    def from_clipboard(**kwargs) -> pd.DataFrame:
        """从剪贴板加载 (方便从 Excel 复制)."""
        return pd.read_clipboard(**kwargs)

    @staticmethod
    def generate_sample(name: str, n_samples=100) -> pd.DataFrame:
        """
        生成示例数据集.

        Parameters
        ----------
        name : str
            数据集名称: "iris", "random_eval", "timeseries", "tsp"
        """
        np.random.seed(42)

        if name == "iris":
            from sklearn.datasets import load_iris
            data = load_iris()
            return pd.DataFrame(data.data, columns=data.feature_names)

        elif name == "random_eval":
            # 评价类数据 (6个方案, 4个指标)
            data = np.random.randint(60, 100, (6, 4))
            return pd.DataFrame(data, columns=["指标A", "指标B", "指标C", "指标D"],
                                index=[f"方案{i+1}" for i in range(6)])

        elif name == "timeseries":
            # 时间序列数据
            t = np.arange(100)
            trend = 0.1 * t
            seasonal = 5 * np.sin(2 * np.pi * t / 12)
            noise = np.random.randn(100) * 2
            values = 50 + trend + seasonal + noise
            return pd.DataFrame({"time": t, "value": values})

        elif name == "ahp_matrix":
            # AHP 判断矩阵
            matrix = np.array([
                [1, 3, 5, 7],
                [1/3, 1, 3, 5],
                [1/5, 1/3, 1, 3],
                [1/7, 1/5, 1/3, 1],
            ])
            return pd.DataFrame(matrix, columns=["价格", "质量", "服务", "外观"],
                                index=["价格", "质量", "服务", "外观"])

        else:
            raise ValueError(f"未知数据集: {name}")
