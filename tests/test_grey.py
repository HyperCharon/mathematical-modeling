"""
测试 grey 模块: GM1N, GreyDecision
"""

import numpy as np
import pytest


class TestGM1N:
    """测试 GM(1,N) 模型."""

    def test_fit_basic(self):
        """基本拟合测试."""
        from mathflow.grey.gm1n import GM1N
        X = np.array([
            [1, 2, 3],
            [2, 3, 5],
            [3, 5, 7],
            [5, 7, 10],
            [7, 10, 14]
        ], dtype=float)
        model = GM1N(X, target_var=0)
        model.fit()
        result = model.result
        assert len(result.fitted_values) == 5
        assert len(result.residuals) == 5

    def test_fit_r2_reasonable(self):
        """R² 应在合理范围内."""
        from mathflow.grey.gm1n import GM1N
        np.random.seed(42)
        X = np.random.rand(10, 3) * 100
        model = GM1N(X, target_var=0)
        model.fit()
        result = model.result
        assert -1 < result.r2 < 1

    def test_predict_after_fit(self):
        """fit 后 predict 应返回正确长度."""
        from mathflow.grey.gm1n import GM1N
        X = np.array([
            [1, 2, 3],
            [2, 3, 5],
            [3, 5, 7],
            [5, 7, 10],
            [7, 10, 14]
        ], dtype=float)
        model = GM1N(X, target_var=0)
        model.fit()
        X_future = np.array([[10, 14, 20]])
        pred = model.predict(X_future)
        assert len(pred) == 1

    def test_predict_before_fit_raises(self):
        """fit 前调用 predict 应抛出异常."""
        from mathflow.grey.gm1n import GM1N
        X = np.array([[1, 2], [3, 4]], dtype=float)
        model = GM1N(X, target_var=0)
        with pytest.raises(RuntimeError):
            model.predict(np.array([[5, 6]]))


class TestGreyDecision:
    """测试灰色决策."""

    def test_grey_relational(self):
        """灰色关联评估测试."""
        from mathflow.grey.grey_decision import GreyDecision
        data = np.array([
            [80, 90, 85],
            [70, 80, 90],
            [90, 85, 75]
        ], dtype=float)
        weights = [0.4, 0.3, 0.3]
        types = [1, 1, 1]  # 全部效益型
        gd = GreyDecision(data, weights, types)
        gd.evaluate("grey_relational")
        result = gd.result
        assert len(result.scores) == 3
        assert len(result.rankings) == 3

    def test_grey_relational_cost_type(self):
        """成本型指标应正确反转."""
        from mathflow.grey.grey_decision import GreyDecision
        data = np.array([
            [100, 50],
            [80, 60],
            [120, 40]
        ], dtype=float)
        weights = [0.5, 0.5]
        types = [-1, 1]  # 第一列成本型
        gd = GreyDecision(data, weights, types)
        gd.evaluate("grey_relational")
        result = gd.result
        assert len(result.scores) == 3

    def test_unknown_method_raises(self):
        """未知方法应抛出异常."""
        from mathflow.grey.grey_decision import GreyDecision
        data = np.array([[1, 2], [3, 4]], dtype=float)
        gd = GreyDecision(data, [0.5, 0.5], [1, 1])
        with pytest.raises(ValueError):
            gd.evaluate("unknown")

    def test_summary(self):
        """summary 应返回字符串."""
        from mathflow.grey.grey_decision import GreyDecision
        data = np.array([[80, 90], [70, 80]], dtype=float)
        gd = GreyDecision(data, [0.5, 0.5], [1, 1])
        gd.evaluate("grey_relational")
        s = gd.summary(labels=["方案A", "方案B"])
        assert isinstance(s, str)
        assert "方案A" in s


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
