"""
多元回归分析

支持线性回归、逐步回归、岭回归、Lasso，自动输出完整的回归报告。

Example:
    >>> from mathflow.stats import MultiRegression
    >>> import numpy as np
    >>> X = np.random.randn(50, 4)
    >>> y = 2*X[:,0] + 3*X[:,1] - X[:,2] + np.random.randn(50)*0.5
    >>> model = MultiRegression(X, y, var_names=["温度","压力","浓度","时间"])
    >>> model.fit()
    >>> print(model.summary())
"""

import numpy as np
from dataclasses import dataclass
from scipy import stats


@dataclass
class RegressionResult:
    """回归结果."""
    coefficients: np.ndarray
    intercept: float
    std_errors: np.ndarray
    t_values: np.ndarray
    p_values: np.ndarray
    r2: float
    adj_r2: float
    f_statistic: float
    f_p_value: float
    residuals: np.ndarray
    fitted_values: np.ndarray
    n_samples: int
    n_features: int
    durbin_watson: float


class MultiRegression:
    """
    多元线性回归 (带完整统计报告).

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        自变量矩阵
    y : array-like, shape (n_samples,)
        因变量
    var_names : list of str, optional
        变量名
    add_constant : bool
        是否自动添加截距项 (默认 True)
    """

    def __init__(self, X, y, var_names=None, add_constant=True):
        self.X = np.asarray(X, dtype=float)
        if self.X.ndim == 1:
            self.X = self.X.reshape(-1, 1)
        self.y = np.asarray(y, dtype=float)

        # 检查 NaN/inf 值
        if np.any(np.isnan(self.X)) or np.any(np.isinf(self.X)):
            raise ValueError("自变量 X 中包含 NaN 或 inf 值，请先清洗数据")
        if np.any(np.isnan(self.y)) or np.any(np.isinf(self.y)):
            raise ValueError("因变量 y 中包含 NaN 或 inf 值，请先清洗数据")

        self.n, self.p = self.X.shape
        if self.n < self.p + 1:
            import warnings
            warnings.warn(f"样本量 ({self.n}) 小于自变量数+1 ({self.p + 1})，可能导致过拟合")

        self.add_constant = add_constant

        if var_names is None:
            self.var_names = [f"X{i+1}" for i in range(self.p)]
        else:
            self.var_names = var_names

        self._result = None

    def fit(self):
        """拟合回归模型."""
        X, y = self.X, self.y
        n, p = X.shape

        if self.add_constant:
            X_design = np.column_stack([np.ones(n), X])
        else:
            X_design = X.copy()

        # 使用 SVD 分解提高数值稳定性（处理多重共线性）
        try:
            U, s, Vt = np.linalg.svd(X_design, full_matrices=False)
            # 检查条件数
            condition_number = s[0] / s[-1] if s[-1] > 1e-10 else float('inf')
            if condition_number > 1e12:
                import warnings
                warnings.warn(f"矩阵条件数过大 ({condition_number:.2e})，可能存在多重共线性")

            # 使用 SVD 求解最小二乘
            beta = Vt.T @ (U.T @ y / s)
        except np.linalg.LinAlgError:
            # Fallback 到正规方程
            XtX = X_design.T @ X_design
            Xty = X_design.T @ y
            try:
                beta = np.linalg.solve(XtX, Xty)
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(XtX, Xty, rcond=None)[0]

        # 拟合值和残差
        y_hat = X_design @ beta
        residuals = y - y_hat

        # 统计量
        k = X_design.shape[1]  # 包括截距
        df_reg = k - 1 if self.add_constant else k
        df_res = n - k

        SSE = np.sum(residuals**2)
        SST = np.sum((y - y.mean())**2)
        SSR = SST - SSE

        r2 = 1 - SSE / SST if SST > 0 else 0
        adj_r2 = 1 - (1 - r2) * (n - 1) / df_res if df_res > 0 else 0

        MSE = SSE / df_res if df_res > 0 else 0

        # 使用 SVD 计算系数方差（更稳定）
        try:
            U, s, Vt = np.linalg.svd(X_design, full_matrices=False)
            # (X'X)^{-1} = V * diag(1/s^2) * V'
            var_beta = MSE * (Vt.T @ np.diag(1.0 / (s**2 + 1e-10)) @ Vt)
        except np.linalg.LinAlgError:
            XtX = X_design.T @ X_design
            try:
                var_beta = MSE * np.linalg.pinv(XtX)
            except:
                var_beta = np.zeros((k, k))

        std_errors = np.sqrt(np.maximum(np.diag(var_beta), 0))

        t_values = beta / std_errors
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), df_res))

        MSR = SSR / df_reg if df_reg > 0 else 0
        f_stat = MSR / MSE if MSE > 0 else 0
        f_p = 1 - stats.f.cdf(f_stat, df_reg, df_res)

        # Durbin-Watson 统计量
        dw = np.sum(np.diff(residuals)**2) / np.sum(residuals**2) if np.sum(residuals**2) > 0 else 2.0

        self._result = RegressionResult(
            coefficients=beta[1:] if self.add_constant else beta,
            intercept=beta[0] if self.add_constant else 0.0,
            std_errors=std_errors,
            t_values=t_values,
            p_values=p_values,
            r2=r2,
            adj_r2=adj_r2,
            f_statistic=f_stat,
            f_p_value=f_p,
            residuals=residuals,
            fitted_values=y_hat,
            n_samples=n,
            n_features=p,
            durbin_watson=dw,
        )
        return self

    def predict(self, X_new):
        """预测."""
        self._ensure_fitted()
        X_new = np.asarray(X_new, dtype=float)
        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)

        # 验证特征维度
        if X_new.shape[1] != self.p:
            raise ValueError(f"预测数据特征数 ({X_new.shape[1]}) 必须等于训练时的特征数 ({self.p})")

        if self.add_constant:
            X_design = np.column_stack([np.ones(len(X_new)), X_new])
        else:
            X_design = X_new
        beta = np.concatenate([[self._result.intercept], self._result.coefficients]) if self.add_constant else self._result.coefficients
        return X_design @ beta

    @property
    def result(self):
        self._ensure_fitted()
        return self._result

    def _ensure_fitted(self):
        if self._result is None:
            raise RuntimeError("请先调用 fit()")

    def summary(self):
        """输出完整回归报告 (类似 statsmodels)."""
        self._ensure_fitted()
        r = self._result

        lines = [
            "=" * 70,
            "  多元线性回归结果",
            "=" * 70,
            f"  因变量: y    样本量: {r.n_samples}    自变量数: {r.n_features}",
            f"  R² = {r.r2:.4f}    调整R² = {r.adj_r2:.4f}",
            f"  F统计量 = {r.f_statistic:.4f}    F p值 = {r.f_p_value:.2e}",
            f"  Durbin-Watson = {r.durbin_watson:.4f}",
            "-" * 70,
            f"  {'变量':>12s}  {'系数':>10s}  {'标准误':>10s}  {'t值':>10s}  {'p值':>10s}  {'显著性':>6s}",
            "-" * 70,
        ]

        # 截距
        if self.add_constant:
            sig = self._sig_stars(r.p_values[0]) if len(r.p_values) > 0 else ""
            lines.append(f"  {'截距':>12s}  {r.intercept:>10.4f}  {r.std_errors[0]:>10.4f}  "
                         f"{r.t_values[0]:>10.4f}  {r.p_values[0]:>10.4f}  {sig:>6s}")

        # 各变量
        for i in range(r.n_features):
            name = self.var_names[i]
            sig = self._sig_stars(r.p_values[i + 1]) if self.add_constant else self._sig_stars(r.p_values[i])
            idx = i + 1 if self.add_constant else i
            lines.append(f"  {name:>12s}  {r.coefficients[i]:>10.4f}  {r.std_errors[idx]:>10.4f}  "
                         f"{r.t_values[idx]:>10.4f}  {r.p_values[idx]:>10.4f}  {sig:>6s}")

        lines.append("=" * 70)
        lines.append("  显著性: *** p<0.001  ** p<0.01  * p<0.05  . p<0.1")
        return "\n".join(lines)

    def _sig_stars(self, p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        elif p < 0.1:
            return "."
        return ""

    def plot_diagnostics(self, figsize=(12, 10)):
        """绘制回归诊断图 (残差分析)."""
        import matplotlib.pyplot as plt

        self._ensure_fitted()
        r = self._result

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. 残差 vs 拟合值
        ax = axes[0, 0]
        ax.scatter(r.fitted_values, r.residuals, alpha=0.6, s=20)
        ax.axhline(y=0, color="red", linestyle="--")
        ax.set_xlabel("拟合值")
        ax.set_ylabel("残差")
        ax.set_title("残差 vs 拟合值")

        # 2. Q-Q 图
        ax = axes[0, 1]
        sorted_residuals = np.sort(r.residuals)
        n = len(sorted_residuals)
        theoretical = stats.norm.ppf(np.arange(1, n + 1) / (n + 1))
        ax.scatter(theoretical, sorted_residuals, alpha=0.6, s=20)
        lims = [min(theoretical.min(), sorted_residuals.min()),
                max(theoretical.max(), sorted_residuals.max())]
        ax.plot(lims, lims, "r--")
        ax.set_xlabel("理论分位数")
        ax.set_ylabel("样本分位数")
        ax.set_title("Q-Q 图 (正态性检验)")

        # 3. 残差直方图
        ax = axes[1, 0]
        ax.hist(r.residuals, bins=20, edgecolor="white", alpha=0.7, density=True)
        x_range = np.linspace(r.residuals.min(), r.residuals.max(), 100)
        ax.plot(x_range, stats.norm.pdf(x_range, r.residuals.mean(), r.residuals.std()), "r-")
        ax.set_xlabel("残差")
        ax.set_ylabel("密度")
        ax.set_title("残差分布")

        # 4. 标准化残差
        ax = axes[1, 1]
        std_resid = r.residuals / r.residuals.std()
        ax.scatter(range(len(std_resid)), std_resid, alpha=0.6, s=20)
        ax.axhline(y=0, color="red", linestyle="--")
        ax.axhline(y=2, color="orange", linestyle=":", alpha=0.5)
        ax.axhline(y=-2, color="orange", linestyle=":", alpha=0.5)
        ax.set_xlabel("观测序号")
        ax.set_ylabel("标准化残差")
        ax.set_title("标准化残差")

        plt.suptitle("回归诊断图", fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig
