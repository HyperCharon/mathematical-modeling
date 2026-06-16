"""
数据清洗工具

Example:
    >>> from mathflow.data import DataCleaner
    >>> import pandas as pd
    >>> df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, None]})
    >>> cleaned = DataCleaner.handle_missing(df, method="mean")
"""

import numpy as np
import pandas as pd


class DataCleaner:
    """数据清洗工具."""

    @staticmethod
    def handle_missing(df, method="mean", fill_value=None):
        """
        处理缺失值.

        Parameters
        ----------
        method : str
            "mean", "median", "mode", "ffill", "bfill", "drop", "value"
        fill_value : optional
            method="value" 时使用的填充值
        """
        df = df.copy()
        if method == "mean":
            return df.fillna(df.mean(numeric_only=True))
        elif method == "median":
            return df.fillna(df.median(numeric_only=True))
        elif method == "mode":
            return df.fillna(df.mode().iloc[0])
        elif method == "ffill":
            return df.ffill()
        elif method == "bfill":
            return df.bfill()
        elif method == "drop":
            return df.dropna()
        elif method == "value":
            return df.fillna(fill_value)
        else:
            raise ValueError(f"未知方法: {method}")

    @staticmethod
    def detect_outliers(df, method="iqr", threshold=1.5):
        """
        检测异常值.

        Returns
        -------
        outlier_mask : DataFrame of bool
        """
        numeric = df.select_dtypes(include=[np.number])
        if method == "iqr":
            Q1 = numeric.quantile(0.25)
            Q3 = numeric.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            return (numeric < lower) | (numeric > upper)
        elif method == "zscore":
            from scipy import stats
            z = np.abs(stats.zscore(numeric, nan_policy="omit"))
            return pd.DataFrame(z > threshold, columns=numeric.columns, index=numeric.index)
        else:
            raise ValueError(f"未知方法: {method}")

    @staticmethod
    def normalize(df, method="minmax"):
        """
        数据标准化.

        Parameters
        ----------
        method : str
            "minmax" (归一化到 [0,1]), "zscore" (标准正态), "maxabs" (最大绝对值)
        """
        numeric = df.select_dtypes(include=[np.number])
        result = df.copy()

        if method == "minmax":
            mins = numeric.min()
            maxs = numeric.max()
            ranges = maxs - mins
            # 对常数列（范围为0）返回0
            ranges = ranges.replace(0, 1)
            result[numeric.columns] = (numeric - mins) / ranges
        elif method == "zscore":
            stds = numeric.std()
            # 对常数列（标准差为0）返回0
            stds = stds.replace(0, 1)
            result[numeric.columns] = (numeric - numeric.mean()) / stds
        elif method == "maxabs":
            max_abs = numeric.abs().max()
            # 对全零列返回0
            max_abs = max_abs.replace(0, 1)
            result[numeric.columns] = numeric / max_abs
        else:
            raise ValueError(f"未知方法: {method}")

        return result

    @staticmethod
    def data_report(df) -> str:
        """生成数据质量报告."""
        lines = ["=" * 60, "  数据质量报告", "=" * 60]
        lines.append(f"  形状: {df.shape[0]} 行 × {df.shape[1]} 列")
        lines.append("-" * 60)

        for col in df.columns:
            missing = df[col].isnull().sum()
            missing_pct = missing / len(df) * 100
            dtype = df[col].dtype
            info = f"  {col:>20s}  类型={str(dtype):>10s}  缺失={missing}({missing_pct:.1f}%)"
            if pd.api.types.is_numeric_dtype(df[col]):
                info += f"  范围=[{df[col].min():.2f}, {df[col].max():.2f}]"
            lines.append(info)

        lines.append("=" * 60)
        return "\n".join(lines)
