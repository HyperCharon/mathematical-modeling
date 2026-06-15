"""ML 模块测试."""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from mathflow.ml import Classifier, ClusterAnalysis, DimensionReduction


class TestClassifier:
    def test_random_forest(self):
        data = load_iris()
        clf = Classifier(data.data, data.target)
        clf.fit(method="random_forest")
        assert clf._result.accuracy > 0.85
        assert clf._result.f1 > 0.85

    def test_logistic(self):
        data = load_iris()
        clf = Classifier(data.data, data.target)
        clf.fit(method="logistic")
        assert clf._result.accuracy > 0.85

    def test_predict(self):
        data = load_iris()
        clf = Classifier(data.data, data.target)
        clf.fit(method="random_forest")
        pred = clf.predict(data.data[:5])
        assert len(pred) == 5

    def test_compare(self):
        data = load_iris()
        clf = Classifier(data.data, data.target)
        df = clf.compare_methods(["logistic", "knn"])
        assert len(df) == 2
        assert "准确率" in df.columns


class TestClusterAnalysis:
    def test_auto_k(self):
        data = load_iris()
        ca = ClusterAnalysis(data.data)
        k = ca.auto_k(max_k=8)
        assert 2 <= k <= 8

    def test_kmeans(self):
        data = load_iris()
        ca = ClusterAnalysis(data.data)
        labels = ca.fit(method="kmeans", n_clusters=3)
        assert len(labels) == 150
        assert len(set(labels)) == 3


class TestDimensionReduction:
    def test_pca(self):
        data = load_iris()
        dr = DimensionReduction(data.data)
        result = dr.fit(n_components=2)
        assert result.transformed.shape == (150, 2)
        assert result.cumulative_variance[-1] > 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
