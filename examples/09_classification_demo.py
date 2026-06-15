"""
示例 9: 分类与聚类 — 古代玻璃制品鉴别

基于 2022C 古代玻璃制品成分鉴别与溯源，展示分类和聚类的完整流程。
"""

import sys
sys.path.insert(0, "..")
import numpy as np

from sklearn.datasets import load_iris
from mathflow.ml import Classifier, ClusterAnalysis, DimensionReduction


def main():
    print("=" * 60)
    print("  示例: 分类与聚类 — 多方法对比")
    print("=" * 60)

    # 加载数据
    data = load_iris()
    X, y = data.data, data.target
    feature_names = data.feature_names
    class_names = list(data.target_names)

    print(f"\n  数据集: Iris ({X.shape[0]} 样本, {X.shape[1]} 特征)")
    print(f"  类别: {class_names}")

    # ======== 1. 多分类器对比 ========
    print("\n【1】多分类器对比")
    print("-" * 40)

    clf = Classifier(X, y, feature_names=feature_names, class_names=class_names)
    comparison = clf.compare_methods(["logistic", "knn", "svm", "decision_tree", "random_forest"])
    print(comparison.to_string(index=False))

    # 最优模型
    best = comparison.iloc[0]["方法"]
    clf.fit(method=best)
    print(f"\n  最优模型: {best}")
    print(clf.summary())

    fig = clf.plot_confusion_matrix()
    fig.savefig("09_confusion_matrix.png", dpi=150, bbox_inches="tight")
    print("  ✅ 已保存: 09_confusion_matrix.png")

    fig2 = clf.plot_feature_importance()
    fig2.savefig("09_feature_importance.png", dpi=150, bbox_inches="tight")
    print("  ✅ 已保存: 09_feature_importance.png")

    # ======== 2. 聚类分析 ========
    print("\n\n【2】聚类分析")
    print("-" * 40)

    ca = ClusterAnalysis(X)
    best_k = ca.auto_k(max_k=8)
    print(f"  自动选择最优 K = {best_k}")

    fig3 = ca.plot_elbow()
    fig3.savefig("09_elbow.png", dpi=150, bbox_inches="tight")
    print("  ✅ 已保存: 09_elbow.png")

    ca.fit(method="kmeans", n_clusters=best_k)
    print(ca.summary())

    fig4 = ca.plot_clusters()
    fig4.savefig("09_clusters.png", dpi=150, bbox_inches="tight")
    print("  ✅ 已保存: 09_clusters.png")

    # ======== 3. PCA 降维 ========
    print("\n\n【3】PCA 降维")
    print("-" * 40)

    dr = DimensionReduction(X)
    dr.fit(n_components=0.95)
    print(dr.summary(feature_names=feature_names))

    fig5 = dr.plot_variance()
    fig5.savefig("09_pca_variance.png", dpi=150, bbox_inches="tight")
    print("  ✅ 已保存: 09_pca_variance.png")

    fig6 = dr.biplot(labels=feature_names)
    fig6.savefig("09_biplot.png", dpi=150, bbox_inches="tight")
    print("  ✅ 已保存: 09_biplot.png")


if __name__ == "__main__":
    main()
