"""
测试 fuzzy 模块: MembershipFunction, FuzzyInference
"""

import numpy as np
import pytest


class TestMembershipFunction:
    """测试隶属函数."""

    def test_triangle_peak(self):
        from mathflow.fuzzy.membership import MembershipFunction
        mf = MembershipFunction()
        x = np.array([0, 1, 2, 3, 4, 5])
        y = mf.triangle(x, 1, 3, 5)
        # 峰值在 b=3 处应为 1
        assert abs(y[3] - 1.0) < 1e-6  # x=3 是峰值
        # a 左侧和 c 右侧应为 0
        assert y[0] == 0  # x=0 < a=1
        assert y[5] == 0  # x=5 = c=5

    def test_triangle_range(self):
        from mathflow.fuzzy.membership import MembershipFunction
        mf = MembershipFunction()
        x = np.linspace(0, 10, 100)
        y = mf.triangle(x, 2, 5, 8)
        assert np.all(y >= 0) and np.all(y <= 1)

    def test_trapezoid_plateau(self):
        from mathflow.fuzzy.membership import MembershipFunction
        mf = MembershipFunction()
        x = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        y = mf.trapezoid(x, 1, 3, 5, 7)
        # 平台区域 [3, 5] 应为 1
        assert abs(y[3] - 1.0) < 1e-6  # x=3
        assert abs(y[4] - 1.0) < 1e-6  # x=4
        assert abs(y[5] - 1.0) < 1e-6  # x=5

    def test_gaussian_peak(self):
        from mathflow.fuzzy.membership import MembershipFunction
        mf = MembershipFunction()
        x = np.array([0, 1, 2, 3, 4, 5])
        y = mf.gaussian(x, 3, 1)
        # 均值处应为 1
        assert abs(y[3] - 1.0) < 1e-6

    def test_gaussian_range(self):
        from mathflow.fuzzy.membership import MembershipFunction
        mf = MembershipFunction()
        x = np.linspace(-5, 5, 100)
        y = mf.gaussian(x, 0, 1)
        assert np.all(y >= 0) and np.all(y <= 1)

    def test_sigmoid_midpoint(self):
        from mathflow.fuzzy.membership import MembershipFunction
        mf = MembershipFunction()
        x = np.array([3.0])
        y = mf.sigmoid(x, 1, 3)
        # x=c 时应为 0.5
        assert abs(y[0] - 0.5) < 1e-6

    def test_z_shape_monotone(self):
        from mathflow.fuzzy.membership import MembershipFunction
        mf = MembershipFunction()
        x = np.linspace(0, 10, 100)
        y = mf.z_shape(x, 3, 7)
        # z-shape 是单调递减的
        assert np.all(np.diff(y) <= 1e-6)

    def test_s_shape_monotone(self):
        from mathflow.fuzzy.membership import MembershipFunction
        mf = MembershipFunction()
        x = np.linspace(0, 10, 100)
        y = mf.s_shape(x, 3, 7)
        # s-shape 是单调递增的
        assert np.all(np.diff(y) >= -1e-6)

    def test_bell_peak(self):
        from mathflow.fuzzy.membership import MembershipFunction
        mf = MembershipFunction()
        x = np.array([3.0])
        y = mf.bell(x, 2, 4, 3)
        # 中心 c=3 处应为 1
        assert abs(y[0] - 1.0) < 1e-6

    def test_all_functions_output_range(self):
        """所有隶属函数输出应在 [0, 1] 范围内."""
        from mathflow.fuzzy.membership import MembershipFunction
        mf = MembershipFunction()
        x = np.linspace(0, 10, 50)

        for func_name in ["triangle", "trapezoid", "gaussian", "sigmoid", "z_shape", "s_shape", "bell"]:
            func = getattr(mf, func_name)
            if func_name == "triangle":
                y = func(x, 2, 5, 8)
            elif func_name == "trapezoid":
                y = func(x, 1, 3, 7, 9)
            elif func_name == "gaussian":
                y = func(x, 5, 1)
            elif func_name == "sigmoid":
                y = func(x, 1, 5)
            elif func_name == "z_shape":
                y = func(x, 3, 7)
            elif func_name == "s_shape":
                y = func(x, 3, 7)
            elif func_name == "bell":
                y = func(x, 2, 4, 5)

            assert np.all(y >= -1e-10), f"{func_name} has negative values"
            assert np.all(y <= 1 + 1e-10), f"{func_name} has values > 1"


class TestFuzzyInference:
    """测试模糊推理系统."""

    def test_basic_inference(self):
        from mathflow.fuzzy.inference import FuzzyInference
        fi = FuzzyInference()

        # 添加输入：温度 (terms 是列表，每个元素是 (name, mf_type, *params))
        fi.add_input("temperature", (0, 40), [
            ("cold", "triangle", 0, 0, 20),
            ("warm", "triangle", 10, 20, 30),
            ("hot", "triangle", 20, 40, 40)
        ])

        # 添加输出：风速
        fi.add_output("fan_speed", (0, 100), [
            ("low", "triangle", 0, 0, 50),
            ("medium", "triangle", 25, 50, 75),
            ("high", "triangle", 50, 100, 100)
        ])

        # 添加规则 (antecedents 是 dict)
        fi.add_rule({"temperature": "cold"}, "fan_speed", "low")
        fi.add_rule({"temperature": "warm"}, "fan_speed", "medium")
        fi.add_rule({"temperature": "hot"}, "fan_speed", "high")

        # 推理 - 返回 dict {输出变量名: 值}
        result = fi.infer({"temperature": 25})
        assert isinstance(result, dict)
        assert "fan_speed" in result
        value = result["fan_speed"]
        assert isinstance(value, (int, float, np.floating))
        assert 0 <= value <= 100

    def test_inference_range(self):
        from mathflow.fuzzy.inference import FuzzyInference
        fi = FuzzyInference()

        fi.add_input("x", (0, 10), [
            ("small", "triangle", 0, 0, 5),
            ("large", "triangle", 5, 10, 10)
        ])

        fi.add_output("y", (0, 100), [
            ("low", "triangle", 0, 0, 50),
            ("high", "triangle", 50, 100, 100)
        ])

        fi.add_rule({"x": "small"}, "y", "low")
        fi.add_rule({"x": "large"}, "y", "high")

        # 测试多个输入值
        for x_val in [1, 3, 5, 7, 9]:
            result = fi.infer({"x": x_val})
            value = result["y"]
            assert 0 <= value <= 100, f"Result {value} out of range for x={x_val}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
