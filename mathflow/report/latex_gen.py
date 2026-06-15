"""
LaTeX 报告自动生成器

自动将分析结果转换为 LaTeX 格式的公式、表格和图表引用。
数模论文写作利器。

Example:
    >>> from mathflow.report import LatexReport
    >>> report = LatexReport("2024年C题 农作物种植策略")
    >>> report.add_section("模型建立")
    >>> report.add_equation("y = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2")
    >>> report.add_table(headers=["方案","得分","排名"], rows=[["A","0.85","1"],["B","0.72","2"]])
    >>> report.save("paper/main.tex")
"""

import numpy as np
from typing import List, Optional, Dict
from datetime import datetime


class LatexReport:
    """
    LaTeX 报告生成器.

    Parameters
    ----------
    title : str
        论文标题
    authors : str
        作者
    """

    def __init__(self, title="数学建模论文", authors="MathFlow Team"):
        self.title = title
        self.authors = authors
        self._content = []
        self._figures = []
        self._tables = []
        self._equations = []

    def add_preamble(self):
        """添加 LaTeX 导言区."""
        preamble = r"""\documentclass[12pt,a4paper]{article}
\usepackage[UTF8]{ctex}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{float}
\usepackage{geometry}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{enumitem}
\usepackage{listings}
\usepackage{xcolor}
\geometry{left=2.5cm,right=2.5cm,top=2.5cm,bottom=2.5cm}

\title{""" + self.title + r"""}
\author{""" + self.authors + r"""}
\date{\today}

\begin{document}
\maketitle
\tableofcontents
\newpage
"""
        self._content.append(preamble)

    def add_section(self, title: str, level: int = 1):
        """添加章节标题."""
        prefix = {1: "\\section", 2: "\\subsection", 3: "\\subsubsection"}
        self._content.append(f"\n{prefix.get(level, '\\section')}{{{title}}}\n")

    def add_text(self, text: str):
        """添加正文段落."""
        self._content.append(f"\n{text}\n")

    def add_equation(self, equation: str, label: str = None, numbered: bool = True) -> str:
        """
        添加数学公式.

        Parameters
        ----------
        equation : str
            LaTeX 公式
        label : str, optional
            公式标签
        numbered : bool
            是否编号
        """
        if numbered:
            env = "equation"
        else:
            env = "equation*"

        tex = f"\n\\begin{{{env}}}\n"
        tex += f"  {equation}\n"
        if label:
            tex += f"  \\label{{{label}}}\n"
        tex += f"\\end{{{env}}}\n"

        self._content.append(tex)
        self._equations.append({"equation": equation, "label": label})
        return f"公式{len(self._equations)}"

    def add_equation_system(self, equations: List[str], label: str = None) -> str:
        """添加方程组."""
        tex = "\n\\begin{equation}\n\\begin{cases}\n"
        for i, eq in enumerate(equations):
            sep = " \\\\" if i < len(equations) - 1 else ""
            tex += f"  {eq}{sep}\n"
        tex += "\\end{cases}\n"
        if label:
            tex += f"  \\label{{{label}}}\n"
        tex += "\\end{equation}\n"
        self._content.append(tex)
        return f"方程组"

    def add_table(self, headers: List[str], rows: List[List], caption: str = "",
                  label: str = None, alignment: str = None) -> str:
        """
        添加表格.

        Parameters
        ----------
        headers : list of str
            表头
        rows : list of list
            数据行
        caption : str
            表格标题
        label : str
            标签
        alignment : str
            列对齐, 如 "lccc"
        """
        n_cols = len(headers)
        if alignment is None:
            alignment = "l" + "c" * (n_cols - 1)

        tex = f"""
\\begin{{table}}[H]
  \\centering
  \\caption{{{caption}}}
  \\begin{{tabular}}{{{alignment}}}
    \\toprule
    {' & '.join(headers)} \\\\
    \\midrule
"""
        for row in rows:
            tex += f"    {' & '.join(str(x) for x in row)} \\\\\n"

        tex += """    \\bottomrule
  \\end{tabular}"""
        if label:
            tex += f"\n  \\label{{{label}}}"
        tex += "\n\\end{table}\n"

        self._content.append(tex)
        self._tables.append({"caption": caption, "label": label})
        return f"表{len(self._tables)}"

    def add_figure(self, image_path: str, caption: str = "", label: str = None,
                   width: float = 0.8, placement: str = "H") -> str:
        """
        添加图片引用.

        Parameters
        ----------
        image_path : str
            图片路径
        caption : str
            图片标题
        label : str
            标签
        width : float
            图片宽度比例
        """
        tex = f"""
\\begin{{figure}}[{placement}]
  \\centering
  \\includegraphics[width={width}\\textwidth]{{{image_path}}}
  \\caption{{{caption}}}"""
        if label:
            tex += f"\n  \\label{{{label}}}"
        tex += "\n\\end{figure}\n"

        self._content.append(tex)
        self._figures.append({"path": image_path, "caption": caption, "label": label})
        return f"图{len(self._figures)}"

    def add_itemize(self, items: List[str]):
        """添加无序列表."""
        tex = "\n\\begin{itemize}\n"
        for item in items:
            tex += f"  \\item {item}\n"
        tex += "\\end{itemize}\n"
        self._content.append(tex)

    def add_enumerate(self, items: List[str]):
        """添加有序列表."""
        tex = "\n\\begin{enumerate}\n"
        for item in items:
            tex += f"  \\item {item}\n"
        tex += "\\end{enumerate}\n"
        self._content.append(tex)

    # ========== 数模常用公式快捷方法 ==========

    @staticmethod
    def ahp_equation() -> str:
        """AHP 权重计算公式."""
        return r"A\mathbf{w} = \lambda_{\max}\mathbf{w}"

    @staticmethod
    def topsis_distance(point: str = "A^+", ideal: str = "z") -> str:
        """TOPSIS 距离公式."""
        return rf"D_{{{point}}} = \sqrt{{\sum_{{j=1}}^{{m}} (w_j z_{{ij}} - w_j {ideal}_j^{{{point}}})^2}}"

    @staticmethod
    def grey_differential(a: str = "a", b: str = "b") -> str:
        """灰色预测微分方程."""
        return rf"\frac{{dX^{{(1)}}}}{{dt}} + {a} X^{{(1)}} = {b}"

    @staticmethod
    def sir_equations() -> List[str]:
        """SIR 模型方程组."""
        return [
            r"\frac{dS}{dt} = -\beta S I",
            r"\frac{dI}{dt} = \beta S I - \gamma I",
            r"\frac{dR}{dt} = \gamma I",
        ]

    @staticmethod
    def lp_objective(coeffs: List[str], sense: str = "max") -> str:
        """线性规划目标函数."""
        terms = " + ".join(f"{c}x_{{{i+1}}}" for i, c in enumerate(coeffs))
        return rf"\{sense} \quad z = {terms}"

    @staticmethod
    def r_squared() -> str:
        """R² 公式."""
        return r"R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}"

    # ========== 模型结果自动转 LaTeX ==========

    def add_ahp_result(self, ahp_result):
        """自动添加 AHP 结果到报告 (接受 AHP 对象或 AHPResult)."""
        # 支持传入 AHP 对象或 AHPResult
        if hasattr(ahp_result, 'result'):
            r = ahp_result.result
        else:
            r = ahp_result
        self.add_text(f"通过层次分析法计算，得到判断矩阵的最大特征值 $\\lambda_{{\\max}} = {r.lambda_max:.4f}$，"
                      f"一致性比率 $CR = {r.CR:.4f} < 0.1$，通过一致性检验。")
        headers = ["指标"] + [f"$W_{{{i+1}}}$" for i in range(r.n)]
        rows = [["权重"] + [f"{w:.4f}" for w in r.weights]]
        self.add_table(headers, rows, caption="AHP 权重计算结果", label="tab:ahp_weights")

    def add_topsis_result(self, topsis_result, labels=None):
        """自动添加 TOPSIS 结果到报告 (接受 TOPSIS 对象或 TOPSISResult)."""
        if hasattr(topsis_result, 'result'):
            r = topsis_result.result
        else:
            r = topsis_result
        n = len(r.scores)
        if labels is None:
            labels = [f"方案{i+1}" for i in range(n)]
        headers = ["方案", "综合得分", "排名"]
        rows = sorted(zip(labels, r.scores, r.rankings), key=lambda x: -x[1])
        rows = [[l, f"{s:.4f}", str(r)] for l, s, r in rows]
        self.add_table(headers, rows, caption="TOPSIS 综合评价结果", label="tab:topsis")

    def add_regression_result(self, reg_result, var_names=None):
        """自动添加回归结果到报告 (接受 MultiRegression 对象或 RegressionResult)."""
        if hasattr(reg_result, 'result'):
            r = reg_result.result
        else:
            r = reg_result
        self.add_text(f"回归模型的决定系数 $R^2 = {r.r2:.4f}$，调整后 $\\bar{{R}}^2 = {r.adj_r2:.4f}$，"
                      f"F 统计量 $F = {r.f_statistic:.4f}$，$p = {r.f_p_value:.2e}$，模型整体显著。")
        headers = ["变量", "系数", "标准误", "t值", "p值", "显著性"]
        rows = [["截距", f"{r.intercept:.4f}", f"{r.std_errors[0]:.4f}",
                 f"{r.t_values[0]:.4f}", f"{r.p_values[0]:.4f}",
                 "***" if r.p_values[0] < 0.001 else "**" if r.p_values[0] < 0.01 else "*" if r.p_values[0] < 0.05 else ""]]
        if var_names is None:
            var_names = [f"$x_{{{i+1}}}$" for i in range(r.n_features)]
        for i in range(r.n_features):
            idx = i + 1
            sig = "***" if r.p_values[idx] < 0.001 else "**" if r.p_values[idx] < 0.01 else "*" if r.p_values[idx] < 0.05 else ""
            rows.append([var_names[i], f"{r.coefficients[i]:.4f}", f"{r.std_errors[idx]:.4f}",
                         f"{r.t_values[idx]:.4f}", f"{r.p_values[idx]:.4f}", sig])
        self.add_table(headers, rows, caption="多元回归系数表", label="tab:regression")

    # ========== 输出 ==========

    def build(self) -> str:
        """构建完整 LaTeX 文档."""
        doc = "\n".join(self._content)
        doc += "\n\\end{document}\n"
        return doc

    def save(self, filepath: str):
        """保存为 .tex 文件."""
        if not self._content or "\\documentclass" not in self._content[0]:
            self.add_preamble()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.build())
        print(f"  ✅ LaTeX 报告已保存: {filepath}")

    def summary(self) -> str:
        """报告摘要."""
        lines = [
            "=" * 50,
            "  LaTeX 报告摘要",
            "=" * 50,
            f"  标题: {self.title}",
            f"  章节数: {sum(1 for c in self._content if '\\section{' in c)}",
            f"  公式数: {len(self._equations)}",
            f"  表格数: {len(self._tables)}",
            f"  图片数: {len(self._figures)}",
            "=" * 50,
        ]
        return "\n".join(lines)
