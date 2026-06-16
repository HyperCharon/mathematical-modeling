"""
测试 data 模块: DataLoader, DataCleaner
"""

import numpy as np
import pandas as pd
import pytest


class TestDataLoader:
    """测试 DataLoader."""

    def test_generate_sample_iris(self):
        from mathflow.data.loader import DataLoader
        df = DataLoader.generate_sample("iris")
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (150, 4)

    def test_generate_sample_random_eval(self):
        from mathflow.data.loader import DataLoader
        df = DataLoader.generate_sample("random_eval")
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] > 0
        assert df.shape[1] > 0

    def test_generate_sample_timeseries(self):
        from mathflow.data.loader import DataLoader
        df = DataLoader.generate_sample("timeseries")
        assert isinstance(df, pd.DataFrame)
        assert "time" in df.columns or len(df.columns) >= 2

    def test_generate_sample_ahp_matrix(self):
        from mathflow.data.loader import DataLoader
        df = DataLoader.generate_sample("ahp_matrix")
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == df.shape[1]  # 方阵
        # 对角线应为 1
        for i in range(len(df)):
            assert abs(df.iloc[i, i] - 1.0) < 1e-6

    def test_generate_sample_unknown_raises(self):
        from mathflow.data.loader import DataLoader
        with pytest.raises(ValueError):
            DataLoader.generate_sample("unknown_dataset")


class TestDataCleaner:
    """测试 DataCleaner."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "A": [1, 2, np.nan, 4, 5],
            "B": [10, 20, 30, np.nan, 50],
            "C": [100, 200, 300, 400, 500]
        })

    def test_handle_missing_mean(self, sample_df):
        from mathflow.data.cleaner import DataCleaner
        result = DataCleaner.handle_missing(sample_df, method="mean")
        assert not result.isna().any().any()

    def test_handle_missing_drop(self, sample_df):
        from mathflow.data.cleaner import DataCleaner
        result = DataCleaner.handle_missing(sample_df, method="drop")
        assert len(result) < len(sample_df)

    def test_handle_missing_ffill(self, sample_df):
        from mathflow.data.cleaner import DataCleaner
        result = DataCleaner.handle_missing(sample_df, method="ffill")
        assert not result.isna().any().any()

    def test_detect_outliers_iqr(self):
        from mathflow.data.cleaner import DataCleaner
        df = pd.DataFrame({"A": [1, 2, 3, 4, 5, 100]})
        result = DataCleaner.detect_outliers(df, method="iqr")
        assert isinstance(result, pd.DataFrame)
        # 100 应该被检测为异常值
        assert result["A"].sum() > 0

    def test_detect_outliers_zscore(self):
        from mathflow.data.cleaner import DataCleaner
        df = pd.DataFrame({"A": [1, 2, 3, 4, 5, 100]})
        result = DataCleaner.detect_outliers(df, method="zscore", threshold=2)
        assert isinstance(result, pd.DataFrame)

    def test_normalize_minmax(self):
        from mathflow.data.cleaner import DataCleaner
        df = pd.DataFrame({"A": [1, 2, 3, 4, 5]})
        result = DataCleaner.normalize(df, method="minmax")
        assert result["A"].min() >= 0
        assert result["A"].max() <= 1

    def test_normalize_zscore(self):
        from mathflow.data.cleaner import DataCleaner
        df = pd.DataFrame({"A": [1, 2, 3, 4, 5]})
        result = DataCleaner.normalize(df, method="zscore")
        assert abs(result["A"].mean()) < 1e-6

    def test_data_report(self, sample_df):
        from mathflow.data.cleaner import DataCleaner
        report = DataCleaner.data_report(sample_df)
        assert isinstance(report, str)
        assert "shape" in report.lower() or "形状" in report or str(sample_df.shape[0]) in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
