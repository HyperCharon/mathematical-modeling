"""
排队论模型 (Queueing Theory)

支持 M/M/1, M/M/c, M/M/1/K 等经典排队模型。

Example:
    >>> from mathflow.simulation import QueueModel
    >>> # M/M/1 排队系统: λ=2, μ=3
    >>> q = QueueModel(arrival_rate=2, service_rate=3, n_servers=1)
    >>> result = q.solve()
    >>> print(f"平均等待时间: {result['W']:.2f}")
"""

import numpy as np
from scipy.special import factorial
from dataclasses import dataclass


class QueueModel:
    """
    排队论模型.

    Parameters
    ----------
    arrival_rate : float
        到达率 λ
    service_rate : float
        服务率 μ (每服务器)
    n_servers : int
        服务器数量 c
    capacity : int, optional
        系统容量 K (None 表示无限)
    """

    def __init__(self, arrival_rate, service_rate, n_servers=1, capacity=None):
        # 验证参数有效性
        if arrival_rate <= 0:
            raise ValueError(f"到达率必须为正数，got {arrival_rate}")
        if service_rate <= 0:
            raise ValueError(f"服务率必须为正数，got {service_rate}")
        if not isinstance(n_servers, int) or n_servers < 1:
            raise ValueError(f"服务器数量必须是正整数，got {n_servers}")
        if capacity is not None and (not isinstance(capacity, int) or capacity < 1):
            raise ValueError(f"系统容量必须是正整数，got {capacity}")

        self.lambda_ = arrival_rate
        self.mu = service_rate
        self.c = n_servers
        self.K = capacity
        self._result = None

    def solve(self):
        """计算排队系统指标."""
        lam = self.lambda_
        mu = self.mu
        c = self.c
        K = self.K

        if K is not None:
            return self._solve_mm1k(lam, mu, K)
        elif c == 1:
            return self._solve_mm1(lam, mu)
        else:
            return self._solve_mmc(lam, mu, c)

    def _solve_mm1(self, lam, mu):
        """M/M/1 模型."""
        rho = lam / mu
        if rho >= 1:
            raise ValueError(f"系统不稳定: ρ={rho:.2f} >= 1, 需要 λ < μ")

        L = rho / (1 - rho)           # 平均队长
        Lq = rho**2 / (1 - rho)       # 平均排队队长
        W = 1 / (mu - lam)            # 平均逗留时间
        Wq = lam / (mu * (mu - lam))  # 平均等待时间

        self._result = {
            "model": "M/M/1",
            "rho": rho, "L": L, "Lq": Lq, "W": W, "Wq": Wq,
            "P0": 1 - rho,
        }
        return self._result

    def _solve_mmc(self, lam, mu, c):
        """M/M/c 模型."""
        rho = lam / (c * mu)
        if rho >= 1:
            raise ValueError(f"系统不稳定: ρ={rho:.2f} >= 1")

        # 计算 P0
        a = lam / mu
        sum_terms = sum((a**k) / factorial(k) for k in range(c))
        last_term = (a**c) / (factorial(c) * (1 - rho))
        P0 = 1.0 / (sum_terms + last_term)

        # Erlang C 公式
        Pc = (a**c) / (factorial(c) * (1 - rho)) * P0

        Lq = Pc * rho / (1 - rho)     # 平均排队队长
        L = Lq + a                     # 平均队长
        Wq = Lq / lam                  # 平均等待时间
        W = Wq + 1 / mu                # 平均逗留时间

        self._result = {
            "model": f"M/M/{c}",
            "rho": rho, "L": L, "Lq": Lq, "W": W, "Wq": Wq,
            "P0": P0, "Pc": Pc,
        }
        return self._result

    def _solve_mm1k(self, lam, mu, K):
        """M/M/1/K 模型 (有限容量)."""
        rho = lam / mu

        if abs(rho - 1) < 1e-10:
            P0 = 1.0 / (K + 1)
            Pn = [P0] * (K + 1)
        else:
            P0 = (1 - rho) / (1 - rho**(K + 1))
            Pn = [P0 * rho**n for n in range(K + 1)]

        # 有效到达率
        lam_eff = lam * (1 - Pn[K])

        L = sum(n * Pn[n] for n in range(K + 1))
        Lq = max(0, L - (1 - P0))

        W = L / lam_eff if lam_eff > 0 else float("inf")
        Wq = Lq / lam_eff if lam_eff > 0 else float("inf")

        self._result = {
            "model": f"M/M/1/{K}",
            "rho": rho, "L": L, "Lq": Lq, "W": W, "Wq": Wq,
            "P0": P0, "P_loss": Pn[K], "lambda_eff": lam_eff,
        }
        return self._result

    def simulate(self, n_arrivals=10000, seed=42):
        """离散事件仿真验证 (支持多服务器)."""
        np.random.seed(seed)
        lam, mu, c = self.lambda_, self.mu, self.c

        inter_arrivals = np.random.exponential(1 / lam, n_arrivals)
        service_times = np.random.exponential(1 / mu, n_arrivals)

        arrival_times = np.cumsum(inter_arrivals)
        # c 个服务器的空闲时间
        server_free_at = np.zeros(c)

        start_service = np.zeros(n_arrivals)
        departure_times = np.zeros(n_arrivals)

        for i in range(n_arrivals):
            # 找最早空闲的服务器
            earliest_server = np.argmin(server_free_at)
            start_service[i] = max(arrival_times[i], server_free_at[earliest_server])
            departure_times[i] = start_service[i] + service_times[i]
            server_free_at[earliest_server] = departure_times[i]

        wait_times = start_service - arrival_times
        system_times = departure_times - arrival_times

        return {
            "sim_Wq": wait_times.mean(),
            "sim_W": system_times.mean(),
            "sim_Lq": (wait_times > 0).mean() * wait_times.mean() * lam,
            "sim_L": system_times.mean() * lam,
        }

    def summary(self):
        if self._result is None:
            raise RuntimeError("请先调用 solve()")
        r = self._result
        lines = ["=" * 50, f"  {r['model']} 排队模型结果", "=" * 50]
        lines.append(f"  到达率 λ = {self.lambda_}")
        lines.append(f"  服务率 μ = {self.mu}")
        if self.c > 1:
            lines.append(f"  服务器数 c = {self.c}")
        lines.append(f"  利用率 ρ = {r['rho']:.4f}")
        lines.append("-" * 50)
        lines.append(f"  平均队长 L  = {r['L']:.4f}")
        lines.append(f"  平均排队长 Lq = {r['Lq']:.4f}")
        lines.append(f"  平均逗留时间 W  = {r['W']:.4f}")
        lines.append(f"  平均等待时间 Wq = {r['Wq']:.4f}")
        if "P_loss" in r:
            lines.append(f"  丢失率 P_loss = {r['P_loss']:.4f}")
        lines.append("=" * 50)
        return "\n".join(lines)
