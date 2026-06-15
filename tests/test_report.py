"""报告模块测试."""

import pytest
from mathflow.report import LatexReport


class TestLatexReport:
    def test_basic(self):
        report = LatexReport("测试论文")
        report.add_preamble()
        report.add_section("引言")
        report.add_text("这是引言部分。")
        report.add_equation("y = ax + b")
        report.add_table(["列1", "列2"], [["1", "2"], ["3", "4"]], caption="测试表")

        content = report.build()
        assert "\\documentclass" in content
        assert "\\section{引言}" in content
        assert "\\begin{equation}" in content
        assert "\\begin{tabular}" in content
        assert "\\end{document}" in content

    def test_equation_system(self):
        report = LatexReport()
        report.add_equation_system(["a = 1", "b = 2", "c = 3"])
        content = report.build()
        assert "\\begin{cases}" in content

    def test_itemize(self):
        report = LatexReport()
        report.add_itemize(["项目1", "项目2"])
        content = report.build()
        assert "\\begin{itemize}" in content
        assert "\\item 项目1" in content

    def test_static_formulas(self):
        assert "lambda" in LatexReport.ahp_equation()
        assert "D_{" in LatexReport.topsis_distance()
        assert "frac{dX" in LatexReport.grey_differential()
        assert len(LatexReport.sir_equations()) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
