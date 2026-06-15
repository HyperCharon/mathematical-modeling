"""
模糊综合评价法 (Fuzzy Comprehensive Evaluation)

处理定性评价问题，将模糊的定性描述定量化。

Example:
    >>> from mathflow.evaluate import FuzzyEvaluation
    >>> import numpy as np
    >>> # 评价因素权重
    >>> weights = [0.3, 0.3, 0.2, 0.2]
    >>> # 模糊评价矩阵 (每行对应一个因素，每列对应一个评语等级)
    >>> R = np.array([
    ...     [0.7, 0.2, 0.1, 0.0],
    ...     [0.5, 0.3, 0.2, 0.0],
    ...     [0.4, 0.4, 0.1, 0.1],
    ...     [0.6, 0.2, 0.1, 0.1],
    ... ])
    >>> fuzzy = FuzzyEvaluation(weights, R, grade_names=["优","良","中","差"])
    >>> fuzzy.fit()
    >>> print(fuzzy.result_grade)
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class FuzzyResult:
    """模糊综合评价结果."""
    fuzzy_vector: np.ndarray   # 综合评价向量 (归一化)
    grade_index: int           # 最大隶属度对应的等级索引
    grade_name: str            # 最大隶属度对应的等级名称
    score: float               # 加权得分 (如果设置了分值)


class FuzzyEvaluation:
    """
    模糊综合评价法.

    Parameters
    ----------
    weights : array-like
        各评价因素的权重
    R : array-like, shape (n_factors, n_grades)
        模糊评价矩阵，R[i][j] 表示第 i 个因素对第 j 个评语的隶属度
    grade_names : list of str, optional
        评语等级名称
    grade_scores : list of float, optional
        各评语等级对应分值，用于计算加权得分
    operator : str
        算子类型: "M(·,+)" 加权平均 (默认), "M(∧,∨)" 主因素突出
    """

    def __init__(self, weights, R, grade_names=None, grade_scores=None, operator="M(·,+)"):
        self.weights = np.asarray(weights, dtype=float)
        self.weights = self.weights / self.weights.sum()
        self.R = np.asarray(R, dtype=float)
        self.operator = operator

        n_grades = self.R.shape[1]
        if grade_names is None:
            grade_names = [f"等级{i+1}" for i in range(n_grades)]
        self.grade_names = grade_names
        self.grade_scores = grade_scores
        self._result = None

    def fit(self):
        """执行模糊综合评价."""
        W = self.weights
        R = self.R

        if self.operator == "M(·,+)":
            # 加权平均型
            B = W @ R
        elif self.operator == "M(∧,∨)":
            # 主因素突出型 (取小取大)
            B = np.zeros(R.shape[1])
            for j in range(R.shape[1]):
                B[j] = np.max(np.minimum(W, R[:, j]))
        else:
            raise ValueError(f"未知算子: {self.operator}")

        # 归一化
        B_sum = B.sum()
        if B_sum > 0:
            B_normalized = B / B_sum
        else:
            B_normalized = B

        # 最大隶属度
        grade_index = np.argmax(B_normalized)
        grade_name = self.grade_names[grade_index]

        # 加权得分
        score = None
        if self.grade_scores is not None:
            score = float(np.dot(B_normalized, self.grade_scores))

        self._result = FuzzyResult(
            fuzzy_vector=B_normalized,
            grade_index=grade_index,
            grade_name=grade_name,
            score=score,
        )
        return self

    @property
    def result_grade(self):
        """返回最大隶属度对应的评语等级."""
        self._ensure_fitted()
        return self._result.grade_name

    @property
    def result(self):
        self._ensure_fitted()
        return self._result

    def _ensure_fitted(self):
        if self._result is None:
            raise RuntimeError("请先调用 fit() 进行计算")

    def plot(self, figsize=(8, 5)):
        """绘制模糊评价结果."""
        import matplotlib.pyplot as plt

        self._ensure_fitted()
        r = self._result
        n = len(r.fuzzy_vector)

        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, n))
        bars = ax.bar(range(n), r.fuzzy_vector, color=colors)
        ax.set_xticks(range(n))
        ax.set_xticklabels(self.grade_names, fontsize=12)
        ax.set_ylabel("隶属度", fontsize=12)
        ax.set_title(f"模糊综合评价结果\n评价等级: {r.grade_name}" +
                     (f", 得分: {r.score:.2f}" if r.score else ""), fontsize=14)

        for bar, v in zip(bars, r.fuzzy_vector):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", fontsize=10)

        plt.tight_layout()
        return fig

    def summary(self):
        """打印结果摘要."""
        self._ensure_fitted()
        r = self._result
        lines = [
            "=" * 50,
            "  模糊综合评价结果",
            "=" * 50,
            f"  算子类型: {self.operator}",
            "-" * 50,
        ]
        for i, name in enumerate(self.grade_names):
            lines.append(f"  {name}: {r.fuzzy_vector[i]:.4f}")
        lines.append("-" * 50)
        lines.append(f"  最终评价: {r.grade_name}")
        if r.score is not None:
            lines.append(f"  综合得分: {r.score:.2f}")
        lines.append("=" * 50)
        return "\n".join(lines)
