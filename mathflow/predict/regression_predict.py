"""
回归预测模型

支持线性回归、多项式回归、岭回归、Lasso回归等。

Example:
    >>> from mathflow.predict import RegressionPredict
    >>> import numpy as np
    >>> X = np.array([1,2,3,4,5,6,7,8]).reshape(-1,1)
    >>> y = np.array([2.1, 3.9, 6.2, 7.8, 10.1, 12.3, 13.8, 16.2])
    >>> model = RegressionPredict(X, y)
    >>> model.fit(method="polynomial", degree=2)
    >>> print(model.r2)
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class RegressionResult:
    """回归预测结果."""
    method: str
    coefficients: np.ndarray
    intercept: float
    r2: float
    rmse: float
    fitted_values: np.ndarray
    residuals: np.ndarray
    equation: str


class RegressionPredict:
    """
    回归预测.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        特征矩阵
    y : array-like, shape (n_samples,)
        目标值
    """

    def __init__(self, X, y):
        self.X = np.asarray(X, dtype=float)
        if self.X.ndim == 1:
            self.X = self.X.reshape(-1, 1)
        self.y = np.asarray(y, dtype=float)
        self._result = None

    def __repr__(self) -> str:
        if self._result is not None:
            return f"RegressionPredict(n_samples={self.X.shape[0]}, n_features={self.X.shape[1]}, R2={self._result.r2:.4f})"
        return f"RegressionPredict(n_samples={self.X.shape[0]}, n_features={self.X.shape[1]})"

    def fit(self, method="linear", degree=2, alpha=1.0):
        """
        拟合回归模型.

        Parameters
        ----------
        method : str
            "linear" (线性), "polynomial" (多项式), "ridge" (岭回归), "lasso" (Lasso)
        degree : int
            多项式阶数 (仅 polynomial)
        alpha : float
            正则化系数 (仅 ridge/lasso)
        """
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        from sklearn.metrics import r2_score, mean_squared_error

        X, y = self.X, self.y

        if method == "polynomial":
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            X_transformed = poly.fit_transform(X)
        else:
            X_transformed = X

        if method == "linear":
            model = LinearRegression()
        elif method == "polynomial":
            model = LinearRegression()
        elif method == "ridge":
            model = Ridge(alpha=alpha)
        elif method == "lasso":
            model = Lasso(alpha=alpha)
        else:
            raise ValueError(f"未知方法: {method}")

        model.fit(X_transformed, y)
        y_pred = model.predict(X_transformed)

        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        # 构建方程字符串
        equation = self._build_equation(model, method, degree)

        self._model = model
        self._poly = poly if method == "polynomial" else None
        self._result = RegressionResult(
            method=method,
            coefficients=model.coef_,
            intercept=model.intercept_,
            r2=r2, rmse=rmse,
            fitted_values=y_pred,
            residuals=y - y_pred,
            equation=equation,
        )
        return self

    def _build_equation(self, model, method, degree):
        """构建方程字符串."""
        coef = model.coef_
        intercept = model.intercept_

        if method in ("linear", "ridge", "lasso"):
            parts = [f"{intercept:.4f}"]
            for i, c in enumerate(coef):
                sign = "+" if c >= 0 else "-"
                parts.append(f" {sign} {abs(c):.4f}·x{i+1}")
            return "y = " + "".join(parts)

        elif method == "polynomial":
            parts = [f"{intercept:.4f}"]
            names = []
            for d in range(1, degree + 1):
                names.append(f"x^{d}")
            for i, c in enumerate(coef):
                if i < len(names):
                    sign = "+" if c >= 0 else "-"
                    parts.append(f" {sign} {abs(c):.4f}·{names[i]}")
            return "y = " + "".join(parts)

        return ""

    def predict(self, X_new):
        """预测新数据."""
        self._ensure_fitted()
        X_new = np.asarray(X_new, dtype=float)
        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)
        if self._poly is not None:
            X_new = self._poly.transform(X_new)
        return self._model.predict(X_new)

    @property
    def r2(self):
        self._ensure_fitted()
        return self._result.r2

    @property
    def rmse(self):
        self._ensure_fitted()
        return self._result.rmse

    def _ensure_fitted(self):
        if self._result is None:
            raise RuntimeError("请先调用 fit() 进行计算")

    def plot(self, figsize=(10, 5)):
        """绘制拟合结果."""
        import matplotlib.pyplot as plt

        self._ensure_fitted()
        r = self._result

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # 左: 拟合曲线 (仅单特征)
        ax = axes[0]
        if self.X.shape[1] == 1:
            order = np.argsort(self.X[:, 0])
            ax.plot(self.X[order, 0], self.y[order], "bo-", label="实际值")
            ax.plot(self.X[order, 0], r.fitted_values[order], "r--", label="拟合值")
        else:
            ax.scatter(self.y, r.fitted_values, alpha=0.6)
            lims = [min(self.y.min(), r.fitted_values.min()),
                    max(self.y.max(), r.fitted_values.max())]
            ax.plot(lims, lims, "r--", label="y=x")
        ax.set_xlabel("实际值")
        ax.set_ylabel("预测值")
        ax.set_title(f"{r.method.title()} 回归\nR²={r.r2:.4f}, RMSE={r.rmse:.4f}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 右: 残差图
        ax2 = axes[1]
        ax2.scatter(r.fitted_values, r.residuals, alpha=0.6)
        ax2.axhline(y=0, color="red", linestyle="--")
        ax2.set_xlabel("拟合值")
        ax2.set_ylabel("残差")
        ax2.set_title("残差分布")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def summary(self):
        """打印结果摘要."""
        self._ensure_fitted()
        r = self._result
        lines = [
            "=" * 50,
            f"  {r.method.title()} 回归预测结果",
            "=" * 50,
            f"  方程: {r.equation}",
            f"  R² = {r.r2:.4f}",
            f"  RMSE = {r.rmse:.4f}",
            "=" * 50,
        ]
        return "\n".join(lines)
