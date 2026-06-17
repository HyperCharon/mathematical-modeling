"""
关键路径法 (CPM) 和计划评审技术 (PERT)

用于项目调度和工期优化，数模 B 题常用。

Example:
    >>> from mathflow.graph import CPM
    >>> cpm = CPM()
    >>> cpm.add_activity("A", duration=3, predecessors=[])
    >>> cpm.add_activity("B", duration=5, predecessors=[])
    >>> cpm.add_activity("C", duration=4, predecessors=["A"])
    >>> cpm.add_activity("D", duration=2, predecessors=["A", "B"])
    >>> cpm.add_activity("E", duration=3, predecessors=["C", "D"])
    >>> result = cpm.solve()
"""

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Activity:
    """活动/工序."""
    name: str
    duration: float
    predecessors: List[str]
    # 计算结果
    ES: float = 0   # 最早开始
    EF: float = 0   # 最早完成
    LS: float = 0   # 最晚开始
    LF: float = 0   # 最晚完成
    TF: float = 0   # 总时差
    FF: float = 0   # 自由时差
    is_critical: bool = False


@dataclass
class CPMResult:
    """CPM 结果."""
    activities: Dict[str, Activity]
    critical_path: List[str]
    project_duration: float
    n_paths: int


class CPM:
    """
    关键路径法 (CPM).

    支持:
    - 前推法 (最早开始/完成)
    - 后推法 (最晚开始/完成)
    - 关键路径识别
    - 总时差和自由时差计算
    """

    def __init__(self):
        self.activities = {}

    def __repr__(self) -> str:
        if hasattr(self, '_result') and self._result is not None:
            return f"CPM(n_activities={len(self.activities)}, duration={self._result.project_duration:.1f})"
        return f"CPM(n_activities={len(self.activities)})"

    def add_activity(self, name: str, duration: float, predecessors: List[str] = None):
        """添加活动."""
        if predecessors is None:
            predecessors = []
        self.activities[name] = Activity(
            name=name, duration=duration, predecessors=predecessors
        )
        return self

    def solve(self) -> CPMResult:
        """求解 CPM."""
        acts = self.activities

        # 前推法: 计算 ES, EF
        # 拓扑排序
        sorted_acts = self._topological_sort()

        for name in sorted_acts:
            act = acts[name]
            if not act.predecessors:
                act.ES = 0
            else:
                act.ES = max(acts[pred].EF for pred in act.predecessors)
            act.EF = act.ES + act.duration

        # 项目总工期
        project_duration = max(act.EF for act in acts.values())

        # 后推法: 计算 LS, LF
        for name in reversed(sorted_acts):
            act = acts[name]
            # 找后继活动
            successors = [a for a in acts.values() if name in a.predecessors]
            if not successors:
                act.LF = project_duration
            else:
                act.LF = min(s.LS for s in successors)
            act.LS = act.LF - act.duration

        # 时差
        for act in acts.values():
            act.TF = act.LS - act.ES
            # 自由时差
            successors = [a for a in acts.values() if act.name in a.predecessors]
            if successors:
                act.FF = min(s.ES for s in successors) - act.EF
            else:
                act.FF = project_duration - act.EF
            act.is_critical = abs(act.TF) < 1e-10

        # 关键路径
        critical_path = [name for name in sorted_acts if acts[name].is_critical]

        self._result = CPMResult(
            activities=acts,
            critical_path=critical_path,
            project_duration=project_duration,
            n_paths=self._count_paths(),
        )
        return self._result

    def _topological_sort(self):
        """拓扑排序."""
        acts = self.activities
        visited = set()
        order = []

        def dfs(name):
            if name in visited:
                return
            visited.add(name)
            for pred in acts[name].predecessors:
                dfs(pred)
            order.append(name)

        for name in acts:
            dfs(name)
        return order

    def _count_paths(self):
        """计算路径数."""
        acts = self.activities
        sorted_acts = self._topological_sort()
        path_count = {name: 0 for name in sorted_acts}
        path_count[sorted_acts[0]] = 1

        for name in sorted_acts:
            for a in acts.values():
                if name in a.predecessors:
                    path_count[a.name] += path_count[name]

        return path_count[sorted_acts[-1]] if sorted_acts else 0

    def plot_gantt(self, figsize=(12, 6)):
        """绘制甘特图."""
        import matplotlib.pyplot as plt

        self._ensure_result()
        r = self._result
        acts = r.activities
        n = len(acts)

        fig, ax = plt.subplots(figsize=figsize)
        y_pos = range(n)

        for i, (name, act) in enumerate(acts.items()):
            color = "red" if act.is_critical else "steelblue"
            ax.barh(i, act.duration, left=act.ES, height=0.6,
                    color=color, alpha=0.8, edgecolor="white")
            ax.text(act.ES + act.duration / 2, i, f"{name}\n({act.duration})",
                    ha="center", va="center", fontsize=9, fontweight="bold")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(list(acts.keys()))
        ax.set_xlabel("时间")
        ax.set_title(f"项目甘特图 (关键路径: {'→'.join(r.critical_path)}, 工期: {r.project_duration})")
        ax.axhline(y=-0.5, color="gray", linewidth=0.5)
        ax.grid(True, alpha=0.3, axis="x")

        # 图例
        from matplotlib.patches import Patch
        ax.legend([Patch(color="red", alpha=0.8), Patch(color="steelblue", alpha=0.8)],
                  ["关键活动", "非关键活动"], loc="lower right")

        plt.tight_layout()
        return fig

    def _ensure_result(self):
        if not hasattr(self, '_result') or self._result is None:
            raise RuntimeError("请先调用 solve()")

    def summary(self):
        self._ensure_result()
        r = self._result
        lines = [
            "=" * 80,
            "  CPM 关键路径法结果",
            "=" * 80,
            f"  项目工期: {r.project_duration}",
            f"  关键路径: {' → '.join(r.critical_path)}",
            "-" * 80,
            f"  {'活动':>6s}  {'工期':>6s}  {'ES':>6s}  {'EF':>6s}  {'LS':>6s}  {'LF':>6s}  "
            f"{'总时差':>6s}  {'自由时差':>6s}  {'关键':>4s}",
            "-" * 80,
        ]
        for name, act in r.activities.items():
            lines.append(
                f"  {name:>6s}  {act.duration:>6.1f}  {act.ES:>6.1f}  {act.EF:>6.1f}  "
                f"{act.LS:>6.1f}  {act.LF:>6.1f}  {act.TF:>6.1f}  {act.FF:>8.1f}  "
                f"{'✅' if act.is_critical else '  ':>4s}"
            )
        lines.append("=" * 80)
        return "\n".join(lines)
