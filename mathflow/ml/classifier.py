"""
分类模型

支持 Logistic 回归、随机森林、SVM、KNN、决策树。
自动输出分类报告、混淆矩阵、特征重要性。

Example:
    >>> from mathflow.ml import Classifier
    >>> from sklearn.datasets import load_iris
    >>> data = load_iris()
    >>> clf = Classifier(data.data, data.target, feature_names=data.feature_names)
    >>> clf.fit(method="random_forest")
    >>> print(clf.summary())
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)


@dataclass
class ClassificationResult:
    """分类结果."""
    method: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray
    feature_importance: Optional[np.ndarray]
    class_report: str
    cv_scores: np.ndarray
    model: object


class Classifier:
    """
    分类器封装.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        特征矩阵
    y : array-like, shape (n_samples,)
        标签
    feature_names : list of str, optional
        特征名
    class_names : list of str, optional
        类别名
    test_size : float
        测试集比例
    """

    def __init__(self, X, y, feature_names=None, class_names=None, test_size=0.2):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y)
        self.feature_names = feature_names or [f"X{i+1}" for i in range(self.X.shape[1])]
        self.class_names = class_names
        self.test_size = test_size
        self._result = None

        # 划分训练/测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42,
            stratify=self.y if len(np.unique(self.y)) > 1 else None
        )

        # 标准化
        self._scaler = StandardScaler()
        self.X_train_s = self._scaler.fit_transform(self.X_train)
        self.X_test_s = self._scaler.transform(self.X_test)

    def fit(self, method="random_forest", **kwargs):
        """
        训练分类器.

        Parameters
        ----------
        method : str
            "logistic", "knn", "svm", "decision_tree", "random_forest", "xgboost"
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

        models = {
            "logistic": lambda: LogisticRegression(max_iter=1000, random_state=42, **kwargs),
            "knn": lambda: KNeighborsClassifier(n_neighbors=kwargs.get("n_neighbors", 5)),
            "svm": lambda: SVC(kernel=kwargs.get("kernel", "rbf"), random_state=42, probability=True),
            "decision_tree": lambda: DecisionTreeClassifier(max_depth=kwargs.get("max_depth", None), random_state=42),
            "random_forest": lambda: RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", None),
                random_state=42
            ),
            "xgboost": lambda: GradientBoostingClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 3),
                random_state=42
            ),
        }

        if method not in models:
            raise ValueError(f"未知方法: {method}, 可选: {list(models.keys())}")

        model = models[method]()

        # 使用标准化数据训练
        model.fit(self.X_train_s, self.y_train)
        y_pred = model.predict(self.X_test_s)

        # 交叉验证
        cv_scores = cross_val_score(model, self.X_train_s, self.y_train, cv=5, scoring="accuracy")

        # 特征重要性
        feature_importance = None
        if hasattr(model, "feature_importances_"):
            feature_importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            feature_importance = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)

        # 分类报告
        target_names = self.class_names if self.class_names else None
        report = classification_report(self.y_test, y_pred, target_names=target_names, zero_division=0)

        self._result = ClassificationResult(
            method=method,
            accuracy=accuracy_score(self.y_test, y_pred),
            precision=precision_score(self.y_test, y_pred, average="weighted", zero_division=0),
            recall=recall_score(self.y_test, y_pred, average="weighted", zero_division=0),
            f1=f1_score(self.y_test, y_pred, average="weighted", zero_division=0),
            confusion_matrix=confusion_matrix(self.y_test, y_pred),
            feature_importance=feature_importance,
            class_report=report,
            cv_scores=cv_scores,
            model=model,
        )
        self._model = model
        return self

    def predict(self, X_new):
        """预测新数据."""
        self._ensure_fitted()
        X_new = self._scaler.transform(np.asarray(X_new, dtype=float))
        return self._model.predict(X_new)

    def predict_proba(self, X_new):
        """预测概率."""
        self._ensure_fitted()
        X_new = self._scaler.transform(np.asarray(X_new, dtype=float))
        return self._model.predict_proba(X_new)

    def _ensure_fitted(self):
        if self._result is None:
            raise RuntimeError("请先调用 fit()")

    def plot_confusion_matrix(self, figsize=(8, 6)):
        """绘制混淆矩阵."""
        import matplotlib.pyplot as plt

        self._ensure_fitted()
        r = self._result
        cm = r.confusion_matrix

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=ax)

        n_classes = cm.shape[0]
        labels = self.class_names if self.class_names else [str(i) for i in range(n_classes)]

        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("预测类别")
        ax.set_ylabel("真实类别")
        ax.set_title(f"混淆矩阵 ({r.method})\n准确率={r.accuracy:.4f}")

        for i in range(n_classes):
            for j in range(n_classes):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=14)

        plt.tight_layout()
        return fig

    def plot_feature_importance(self, top_n=None, figsize=(10, 6)):
        """绘制特征重要性."""
        import matplotlib.pyplot as plt

        self._ensure_fitted()
        r = self._result
        if r.feature_importance is None:
            raise ValueError("该模型不支持特征重要性")

        importance = r.feature_importance
        names = self.feature_names
        n = len(importance)

        if top_n is None:
            top_n = min(n, 15)

        order = np.argsort(-importance)[:top_n]

        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(order)))
        ax.barh(range(len(order)), importance[order], color=colors)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([names[i] for i in order])
        ax.set_xlabel("重要性")
        ax.set_title(f"特征重要性 ({r.method})")
        ax.invert_yaxis()
        plt.tight_layout()
        return fig

    def compare_methods(self, methods=None):
        """比较多种分类方法."""
        if methods is None:
            methods = ["logistic", "knn", "svm", "decision_tree", "random_forest"]

        results = []
        for method in methods:
            try:
                self.fit(method=method)
                results.append({
                    "方法": method,
                    "准确率": self._result.accuracy,
                    "精确率": self._result.precision,
                    "召回率": self._result.recall,
                    "F1": self._result.f1,
                    "CV均值": self._result.cv_scores.mean(),
                    "CV标准差": self._result.cv_scores.std(),
                })
            except Exception as e:
                print(f"  {method} 失败: {e}")

        df = pd.DataFrame(results).sort_values("F1", ascending=False)
        return df

    def summary(self):
        self._ensure_fitted()
        r = self._result
        lines = [
            "=" * 60,
            f"  分类结果 ({r.method})",
            "=" * 60,
            f"  准确率: {r.accuracy:.4f}",
            f"  精确率: {r.precision:.4f}",
            f"  召回率: {r.recall:.4f}",
            f"  F1 分数: {r.f1:.4f}",
            f"  交叉验证: {r.cv_scores.mean():.4f} ± {r.cv_scores.std():.4f}",
            "-" * 60,
            "  分类报告:",
            r.class_report,
            "=" * 60,
        ]
        return "\n".join(lines)
