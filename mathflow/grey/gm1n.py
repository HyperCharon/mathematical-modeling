"""
GM(1,N) 灰色多变量预测模型

适用于多因素影响的预测问题，比 GM(1,1) 考虑更多因素。

Example:
    >>> from mathflow.grey import GM1N
    >>> import numpy as np
    >>> # 3个变量, 5个时间点
    >>> X = np.array([
    ...     [100, 50, 30],
    ...     [110, 55, 35],
    ...     [125, 60, 38],
    ...     [140, 68, 42],
    ...     [155, 75, 48],
    ... ])
    >>> gm = GM1N(X)
    >>> result = gm.fit()
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class GM1NResult:
    """GM(1,N) 结果."""
    coefficients: np.ndarray   # 发展系数和作用量
    fitted_values: np.ndarray  # 拟合值 (还原后)
    residuals: np.ndarray
    relative_errors: np.ndarray
    r2: float


class GM1N:
    """
    GM(1,N) 灰色多变量预测模型.

    Parameters
    ----------
    X : array-like, shape (n_times, n_vars)
        数据矩阵，每行是一个时间点，每列是一个变量
    target_var : int
        目标变量的列索引 (默认最后一列)
    """

    def __init__(self, X, target_var=None):
        self.X = np.asarray(X, dtype=float)
        self.n_times, self.n_vars = self.X.shape
        self.target_var = target_var if target_var is not None else self.n_vars - 1
        self._result = None

    def __repr__(self) -> str:
        return f"GM1N(n_times={self.n_times}, n_vars={self.n_vars})"

    def fit(self) -> GM1NResult:
        """拟合 GM(1,N) 模型."""
        X = self.X
        n, m = X.shape

        # Step 1: 累加生成序列 (AGO)
        X1 = np.cumsum(X, axis=0)

        # Step 2: 紧邻均值生成
        Z1 = 0.5 * (X1[:-1, self.target_var] + X1[1:, self.target_var])

        # Step 3: 构建数据矩阵
        # dx1/dt + a*x1 = b2*x2 + b3*x3 + ... + bm*xm
        B = np.zeros((n - 1, m))
        B[:, 0] = -Z1
        for j in range(1, m):
            if j != self.target_var:
                B[:, j] = X1[1:, j]
            else:
                B[:, j] = 0  # Target variable excluded from driving forces

        Y = X[1:, self.target_var]

        # Step 4: 最小二乘求解
        params = np.linalg.lstsq(B, Y, rcond=None)[0]
        a = params[0]
        b = params[1:]

        # Step 5: 求解微分方程
        X1_hat = np.zeros((n, m))
        X1_hat[0] = X1[0]

        for k in range(1, n):
            # 简化的数值解
            sum_bx = sum(b[j-1] * X1[k-1, j] for j in range(1, m) if j != self.target_var)
            if abs(a) < 1e-10:
                # a 接近零时使用线性近似
                X1_hat[k, self.target_var] = X1[0, self.target_var] + sum_bx * k
            else:
                # 防止指数溢出
                exp_term = np.clip(-a * k, -500, 500)
                X1_hat[k, self.target_var] = (X1[0, self.target_var] - sum_bx / a) * np.exp(exp_term) + sum_bx / a

        # Step 6: 累减还原
        X0_hat = np.zeros_like(X)
        X0_hat[0] = X[0]
        X0_hat[1:, self.target_var] = np.diff(X1_hat[:, self.target_var])
        # 其他变量直接用原始值
        for j in range(m):
            if j != self.target_var:
                X0_hat[:, j] = X[:, j]

        # 残差
        residuals = X[:, self.target_var] - X0_hat[:, self.target_var]
        rel_errors = np.abs(residuals) / np.where(X[:, self.target_var] != 0, X[:, self.target_var], 1)

        # R²
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((X[:, self.target_var] - X[:, self.target_var].mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        self._result = GM1NResult(
            coefficients=params,
            fitted_values=X0_hat,
            residuals=residuals,
            relative_errors=rel_errors,
            r2=r2,
        )
        self._a = a
        self._b = b
        return self._result

    def predict(self, X_future: np.ndarray) -> np.ndarray:
        """用已知的其他变量值预测目标变量."""
        self._ensure_fitted()
        a = self._a
        b = self._b
        X_future = np.asarray(X_future, dtype=float)
        n = len(X_future)

        predictions = np.zeros(n)
        X1_last = self.X[:, self.target_var].sum()

        for k in range(n):
            sum_bx = sum(b[j] * X_future[k, j] for j in range(len(b)))  # j not j+1
            if abs(a) < 1e-10:
                X1_new = X1_last + sum_bx
            else:
                exp_term = np.clip(-a * (k + 1), -500, 500)
                X1_new = (X1_last - sum_bx / a) * np.exp(exp_term) + sum_bx / a
            predictions[k] = X1_new - X1_last
            X1_last = X1_new

        return predictions

    @property
    def result(self):
        self._ensure_fitted()
        return self._result

    def _ensure_fitted(self):
        if self._result is None:
            raise RuntimeError("请先调用 fit()")

    def summary(self):
        self._ensure_fitted()
        r = self._result
        lines = [
            "=" * 50,
            "  GM(1,N) 灰色多变量预测结果",
            "=" * 50,
            f"  变量数: {self.n_vars}",
            f"  时间点数: {self.n_times}",
            f"  发展系数 a = {self._a:.6f}",
            f"  R² = {r.r2:.4f}",
            "-" * 50,
            "  平均相对误差: {:.2%}".format(np.mean(r.relative_errors[1:])),
            "-" * 50,
            "  系数:",
        ]
        for i, coef in enumerate(r.coefficients):
            name = "a" if i == 0 else f"b{i}"
            lines.append(f"    {name} = {coef:.6f}")
        lines.append("=" * 50)
        return "\n".join(lines)
