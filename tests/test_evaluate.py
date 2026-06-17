"""评价类模型测试."""

import numpy as np
import pytest
from mathflow.evaluate import AHP, TOPSIS, EntropyWeight, CRITIC, GreyRelationalAnalysis, FuzzyEvaluation, PROMETHEE, RSR


class TestAHP:
    def test_basic(self):
        ahp = AHP()
        ahp.set_matrix([
            [1,   3,   5],
            [1/3, 1,   3],
            [1/5, 1/3, 1]
        ])
        ahp.fit()
        assert len(ahp.weights) == 3
        assert abs(ahp.weights.sum() - 1.0) < 1e-6
        assert ahp.is_consistent  # CR < 0.10

    def test_geometric_method(self):
        ahp = AHP(method="geometric")
        ahp.set_matrix([
            [1,   3,   5],
            [1/3, 1,   3],
            [1/5, 1/3, 1]
        ])
        ahp.fit()
        assert len(ahp.weights) == 3
        assert ahp.weights[0] > ahp.weights[1] > ahp.weights[2]


class TestTOPSIS:
    def test_basic(self):
        data = np.array([
            [80, 90, 85, 70],
            [70, 80, 90, 80],
            [90, 85, 80, 75],
            [85, 75, 88, 85],
        ])
        topsis = TOPSIS(data, weights=[0.3, 0.25, 0.25, 0.2])
        topsis.fit()
        assert len(topsis.scores) == 4
        assert all(0 <= s <= 1 for s in topsis.scores)
        assert set(topsis.rankings) == {1, 2, 3, 4}

    def test_cost_type(self):
        data = np.array([
            [100, 50],
            [200, 30],
            [150, 40],
        ])
        topsis = TOPSIS(data, weights=[0.5, 0.5], types=[-1, 1])
        topsis.fit()
        assert len(topsis.scores) == 3


class TestEntropyWeight:
    def test_basic(self):
        data = np.array([
            [80, 90, 85],
            [70, 80, 90],
            [90, 85, 80],
            [85, 75, 88],
        ])
        ew = EntropyWeight(data)
        ew.fit()
        assert abs(ew.weights.sum() - 1.0) < 1e-6
        assert all(w > 0 for w in ew.weights)


class TestCRITIC:
    def test_basic(self):
        data = np.array([
            [80, 90, 85],
            [70, 80, 90],
            [90, 85, 80],
            [85, 75, 88],
        ])
        critic = CRITIC(data)
        critic.fit()
        assert abs(critic.weights.sum() - 1.0) < 1e-6


class TestGRA:
    def test_basic(self):
        ref = np.array([1.0, 1.1, 1.2, 1.3, 1.4])
        comp = np.array([
            [1.0, 0.9, 1.1, 1.2, 1.3],
            [0.8, 1.0, 1.0, 1.1, 1.2],
        ])
        gra = GreyRelationalAnalysis(ref, comp)
        gra.fit()
        assert len(gra.correlations) == 2
        assert all(0 < c < 1 for c in gra.correlations)


class TestFuzzyEvaluation:
    def test_basic(self):
        weights = [0.3, 0.3, 0.2, 0.2]
        R = np.array([
            [0.7, 0.2, 0.1, 0.0],
            [0.5, 0.3, 0.2, 0.0],
            [0.4, 0.4, 0.1, 0.1],
            [0.6, 0.2, 0.1, 0.1],
        ])
        fuzzy = FuzzyEvaluation(weights, R, grade_names=["优", "良", "中", "差"])
        fuzzy.fit()
        assert fuzzy.result_grade in ["优", "良", "中", "差"]
        assert abs(fuzzy.result.fuzzy_vector.sum() - 1.0) < 1e-6


class TestPROMETHEE:
    def test_basic(self):
        data = np.array([[80, 90, 85], [70, 80, 90], [90, 85, 75]])
        p = PROMETHEE(data, weights=[1/3, 1/3, 1/3], types=[1, 1, -1])
        result = p.fit()
        assert result is not None
        assert len(result.rankings) == 3
        assert len(result.phi_net) == 3

    def test_repr(self):
        data = np.array([[80, 90], [70, 80]])
        p = PROMETHEE(data, weights=[0.5, 0.5])
        assert "PROMETHEE" in repr(p)


class TestRSR:
    def test_basic(self):
        data = np.array([[80, 90, 85], [70, 80, 90], [90, 85, 75]])
        r = RSR(data, types=[1, 1, 1])
        r.fit()
        assert len(r.rankings) == 3
        assert len(r.rsr_values) == 3

    def test_repr(self):
        data = np.array([[80, 90], [70, 80]])
        r = RSR(data)
        assert "RSR" in repr(r)


class TestEdgeCases:
    def test_topsis_single_sample(self):
        """单样本 TOPSIS 应能运行."""
        data = np.array([[80, 90, 85]])
        t = TOPSIS(data, types=[1, 1, 1])
        t.fit()
        assert len(t.rankings) == 1

    def test_topsis_constant_column(self):
        """常数列 TOPSIS 应能运行."""
        data = np.array([[80, 50, 85], [80, 60, 90], [80, 70, 75]])
        t = TOPSIS(data, types=[1, 1, 1])
        t.fit()
        assert len(t.rankings) == 3

    def test_ahp_2x2(self):
        """2x2 AHP 矩阵."""
        ahp = AHP()
        ahp.set_matrix([[1, 3], [1/3, 1]])
        ahp.fit()
        assert len(ahp.weights) == 2
        assert ahp.is_consistent

    def test_entropy_weight_constant_data(self):
        """常数数据熵权法应能处理."""
        data = np.array([[5, 5, 5], [5, 5, 5]])
        ew = EntropyWeight(data)
        # 常数数据可能产生 NaN，不应崩溃
        try:
            ew.fit()
        except (ValueError, RuntimeWarning):
            pass  # 可接受

    def test_gra_single_compare(self):
        """单比较序列 GRA."""
        ref = np.array([1, 2, 3])
        comp = np.array([[1.1, 2.1, 3.1]])
        gra = GreyRelationalAnalysis(ref, comp)
        gra.fit()
        assert len(gra.correlations) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
