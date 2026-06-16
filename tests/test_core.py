"""
测试 core 模块: Config, validators
"""

import numpy as np
import pandas as pd
import pytest


class TestConfig:
    """测试 Config 数据类."""

    def test_default_values(self):
        from mathflow.core.config import Config
        cfg = Config()
        assert cfg.epsilon == 1e-10
        assert cfg.random_seed == 42
        assert cfg.figure_dpi == 150
        assert cfg.font_size == 12
        assert cfg.decimal_places == 4
        assert cfg.ahp_cr_threshold == 0.1

    def test_global_config_singleton(self):
        from mathflow.core.config import config
        from mathflow.core.config import Config
        default = Config()
        assert config.epsilon == default.epsilon
        assert config.random_seed == default.random_seed


class TestValidators:
    """测试验证函数."""

    def test_validate_matrix_list(self):
        from mathflow.core.validators import validate_matrix
        result = validate_matrix([[1, 2], [3, 4]])
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)

    def test_validate_matrix_dataframe(self):
        from mathflow.core.validators import validate_matrix
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = validate_matrix(df)
        assert isinstance(result, np.ndarray)

    def test_validate_matrix_nan_raises(self):
        from mathflow.core.validators import validate_matrix
        with pytest.raises(ValueError):
            validate_matrix([[1, np.nan], [3, 4]])

    def test_validate_positive_valid(self):
        from mathflow.core.validators import validate_positive
        result = validate_positive([[1, 2], [3, 4]])
        assert isinstance(result, np.ndarray)

    def test_validate_positive_zero_raises(self):
        from mathflow.core.validators import validate_positive
        with pytest.raises(ValueError):
            validate_positive([[0, 2], [3, 4]])

    def test_validate_positive_negative_raises(self):
        from mathflow.core.validators import validate_positive
        with pytest.raises(ValueError):
            validate_positive([[-1, 2], [3, 4]])

    def test_validate_square_matrix_valid(self):
        from mathflow.core.validators import validate_square_matrix
        result = validate_square_matrix([[1, 2], [3, 4]])
        assert result.shape == (2, 2)

    def test_validate_square_matrix_non_square_raises(self):
        from mathflow.core.validators import validate_square_matrix
        with pytest.raises(ValueError):
            validate_square_matrix([[1, 2, 3], [4, 5, 6]])

    def test_ensure_numpy_decorator(self):
        from mathflow.core.validators import ensure_numpy

        @ensure_numpy
        def my_func(data):
            assert isinstance(data, np.ndarray)
            return data.sum()

        df = pd.DataFrame({"a": [1, 2, 3]})
        result = my_func(df)
        assert result == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
