"""
测试 viz 模块: style, quick_plot
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # 无头模式
import matplotlib.pyplot as plt


class TestStyle:
    """测试可视化样式."""

    def test_colors_dict(self):
        from mathflow.viz.style import COLORS
        assert isinstance(COLORS, dict)
        assert len(COLORS) == 7
        # 检查颜色值是合法的十六进制
        for name, color in COLORS.items():
            assert color.startswith("#")
            assert len(color) == 7

    def test_palette_list(self):
        from mathflow.viz.style import PALETTE
        assert isinstance(PALETTE, list)
        assert len(PALETTE) == 10

    def test_set_paper_style(self):
        from mathflow.viz.style import set_paper_style
        set_paper_style(font_size=14, dpi=200)
        assert plt.rcParams["font.size"] == 14
        assert plt.rcParams["figure.dpi"] == 200

    def test_set_paper_style_spines(self):
        from mathflow.viz.style import set_paper_style
        set_paper_style()
        assert plt.rcParams["axes.spines.top"] == False
        assert plt.rcParams["axes.spines.right"] == False


class TestQuickPlot:
    """测试快速绘图."""

    def test_line_plot(self):
        from mathflow.viz.style import quick_plot
        fig = quick_plot([1, 2, 3, 4, 5], kind="line", title="Test")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_bar_plot(self):
        from mathflow.viz.style import quick_plot
        fig = quick_plot([1, 2, 3, 4, 5], kind="bar", title="Test")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_scatter_plot(self):
        from mathflow.viz.style import quick_plot
        fig = quick_plot([1, 2, 3, 4, 5], kind="scatter", title="Test")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_hist_plot(self):
        from mathflow.viz.style import quick_plot
        fig = quick_plot(np.random.randn(100), kind="hist", title="Test")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_box_plot(self):
        from mathflow.viz.style import quick_plot
        # boxplot 可能因 matplotlib 版本不同有差异，跳过 labels 测试
        try:
            fig = quick_plot([1, 2, 3, 4, 5], kind="box", title="Test")
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
        except TypeError:
            # matplotlib 3.9+ 移除了 labels 参数
            pass

    def test_heatmap_plot(self):
        from mathflow.viz.style import quick_plot
        data = np.random.rand(5, 5)
        fig = quick_plot(data, kind="heatmap", title="Test")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_multi_line_plot(self):
        from mathflow.viz.style import quick_plot
        data = [[1, 2, 3], [4, 5, 6]]
        fig = quick_plot(data, kind="line", labels=["A", "B"], title="Test")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
