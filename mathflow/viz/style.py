"""
论文风格可视化主题

Example:
    >>> from mathflow.viz import set_paper_style
    >>> set_paper_style()
    >>> import matplotlib.pyplot as plt
    >>> plt.plot([1,2,3], [1,4,9])
    >>> plt.show()
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from mathflow.core.config import config

# 数模论文常用配色
COLORS = {
    "blue": "#3B82F6",
    "red": "#EF4444",
    "green": "#10B981",
    "orange": "#F59E0B",
    "purple": "#8B5CF6",
    "gray": "#6B7280",
    "dark": "#1F2937",
}

PALETTE = ["#3B82F6", "#EF4444", "#10B981", "#F59E0B", "#8B5CF6",
           "#EC4899", "#14B8A6", "#F97316", "#6366F1", "#84CC16"]


def set_paper_style(font_size=None, dpi=None):
    """设置论文风格 (中文友好)."""
    if font_size is None:
        font_size = config.font_size
    if dpi is None:
        dpi = config.figure_dpi
    plt.rcParams.update({
        "figure.dpi": dpi,
        "font.size": font_size,
        "axes.titlesize": font_size + 2,
        "axes.labelsize": font_size + 1,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "figure.figsize": (8, 6),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
    })
    # 尝试设置中文字体
    for font in ["SimHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC", "Microsoft YaHei"]:
        try:
            plt.rcParams["font.sans-serif"] = [font] + plt.rcParams["font.sans-serif"]
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False


def quick_plot(data, kind="line", title="", xlabel="", ylabel="", labels=None, figsize=(8, 6)):
    """
    快速绘图.

    Parameters
    ----------
    data : array-like or list of array-like
        数据
    kind : str
        "line", "bar", "scatter", "hist", "box", "heatmap"
    """
    import numpy as np

    set_paper_style()
    fig, ax = plt.subplots(figsize=figsize)

    if kind == "line":
        if isinstance(data, (list, tuple)) and isinstance(data[0], (list, np.ndarray)):
            for i, d in enumerate(data):
                lbl = labels[i] if labels else f"系列{i+1}"
                ax.plot(d, label=lbl)
            ax.legend()
        else:
            ax.plot(data)

    elif kind == "bar":
        x = np.arange(len(data))
        ax.bar(x, data)
        if labels:
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")

    elif kind == "scatter":
        if isinstance(data, (list, tuple)) and len(data) >= 2:
            ax.scatter(data[0], data[1])

    elif kind == "hist":
        ax.hist(data, bins=30, edgecolor="white", alpha=0.7)

    elif kind == "box":
        ax.boxplot(data, labels=labels)

    elif kind == "heatmap":
        im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
        plt.colorbar(im, ax=ax)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    return fig
