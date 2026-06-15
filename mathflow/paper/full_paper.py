"""
完整论文模板生成器

一键生成符合数模国赛规范的完整论文框架，填入分析结果即可。

Example:
    >>> from mathflow.paper import FullPaper
    >>> paper = FullPaper("2024年C题 农作物种植策略优化")
    >>> paper.set_background("...")
    >>> paper.add_sub_problem("最优种植方案", "线性规划")
    >>> paper.add_sub_problem("风险分析", "蒙特卡洛模拟")
    >>> paper.generate("paper.tex")
"""

from typing import List, Dict, Optional
from mathflow.paper.abstract_gen import AbstractGenerator
from mathflow.paper.section_gen import SectionGenerator
from mathflow.paper.model_eval import ModelEvaluator


class FullPaper:
    """
    完整论文模板生成器.

    自动生成包含所有标准章节的 LaTeX 论文框架。
    """

    def __init__(self, title: str, year: int = 2026):
        self.title = title
        self.year = year
        self.abstract_gen = AbstractGenerator(title)
        self.section_gen = SectionGenerator()
        self.evaluators = []
        self.sub_problems = []
        self.sensitivity_params = []
        self._background = ""
        self._data_source = ""

    def set_background(self, background: str):
        """设置问题背景."""
        self._background = background
        return self

    def set_data_source(self, source: str):
        """设置数据来源."""
        self._data_source = source
        return self

    def add_sub_problem(self, description: str, method: str,
                        result: str = "", model_name: str = ""):
        """添加子问题."""
        self.sub_problems.append({
            "description": description,
            "method": method,
            "result": result,
            "model_name": model_name,
        })
        self.abstract_gen.add_problem_solution(description, method, result or "待填入", model_name)
        return self

    def add_assumption(self, text: str):
        """添加模型假设."""
        self.section_gen.add_assumption(text)
        return self

    def add_symbol(self, symbol: str, meaning: str, unit: str = ""):
        """添加符号."""
        self.section_gen.add_symbol(symbol, meaning, unit)
        return self

    def add_sensitivity_param(self, name: str, base_value: float, description: str = ""):
        """添加灵敏度分析参数."""
        self.sensitivity_params.append({
            "name": name, "base_value": base_value, "description": description,
        })
        return self

    def set_keywords(self, keywords: List[str]):
        """设置关键词."""
        self.abstract_gen.set_keywords(keywords)
        return self

    def generate(self, filepath: str = None) -> str:
        """
        生成完整论文.

        Parameters
        ----------
        filepath : str, optional
            保存路径 (如 "paper.tex")
        """
        lines = []

        # LaTeX 导言
        lines.append(self._generate_preamble())
        lines.append("")

        # 摘要
        lines.append(self._generate_abstract_section())
        lines.append("")

        # 正文
        lines.append("\\section{问题重述}")
        lines.append(self._generate_problem_restatement())
        lines.append("")

        lines.append("\\section{问题分析}")
        lines.append(self._generate_problem_analysis())
        lines.append("")

        lines.append("\\section{模型假设}")
        lines.append(self._generate_assumptions())
        lines.append("")

        lines.append("\\section{符号说明}")
        lines.append(self._generate_symbols())
        lines.append("")

        # 各问题的模型建立与求解
        for i, sp in enumerate(self.sub_problems, 1):
            lines.append(f"\\section{{问题{i}的模型建立与求解}}")
            lines.append(self._generate_sub_problem_section(i, sp))
            lines.append("")

        # 灵敏度分析
        if self.sensitivity_params:
            lines.append("\\section{灵敏度分析}")
            lines.append(self._generate_sensitivity_section())
            lines.append("")

        # 模型评价
        lines.append("\\section{模型评价与推广}")
        lines.append(self._generate_evaluation_section())
        lines.append("")

        # 参考文献
        lines.append("\\section{参考文献}")
        self.section_gen.add_common_references()
        lines.append(self.section_gen.generate_references())

        # 附录
        lines.append("\\section{附录}")
        lines.append("\\subsection{代码}")
        lines.append("% 在此粘贴核心代码")
        lines.append("")
        lines.append("\\subsection{数据}")
        lines.append("% 在此粘贴数据表格")

        lines.append("")
        lines.append("\\end{document}")

        content = "\n".join(lines)

        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  ✅ 论文模板已保存: {filepath}")

        return content

    def _generate_preamble(self) -> str:
        return r"""\documentclass[12pt,a4paper]{article}
\usepackage[UTF8]{ctex}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{float}
\usepackage{geometry}
\usepackage{caption}
\usepackage{enumitem}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{fancyhdr}
\usepackage{titlesec}
\geometry{left=2.5cm,right=2.5cm,top=2.5cm,bottom=2.5cm}

% 页眉页脚
\pagestyle{fancy}
\fancyhf{}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0pt}

% 标题格式
\titleformat{\section}{\large\bfseries}{\thesection}{1em}{}
\titleformat{\subsection}{\normalsize\bfseries}{\thesubsection}{1em}{}

\title{\textbf{""" + self.title + r"""}}
\date{}

\begin{document}
\maketitle"""

    def _generate_abstract_section(self) -> str:
        abstract_text = self.abstract_gen.generate("standard")
        # 将摘要内容中的换行转为LaTeX段落
        abstract_paras = abstract_text.split("\n\n")

        lines = [
            "\\begin{abstract}",
        ]
        for para in abstract_paras:
            if para.strip():
                lines.append(para.strip())
                lines.append("")
        lines.append("\\end{abstract}")
        return "\n".join(lines)

    def _generate_problem_restatement(self) -> str:
        return (
            f"本文研究的是{self.year}年{self.title}问题。\n\n"
            f"{self._background if self._background else '（在此用自己的语言重述问题，不要照抄原题）'}\n\n"
            f"数据来源：{self._data_source if self._data_source else '（在此说明数据来源）'}"
        )

    def _generate_problem_analysis(self) -> str:
        sub_desc = []
        for i, sp in enumerate(self.sub_problems, 1):
            sub_desc.append({"description": sp["description"], "approach": sp["method"]})
        return self.section_gen.generate_problem_analysis(
            problem=f"针对{self.title}问题，通过深入分析，我们将其分解为{len(self.sub_problems)}个子问题。",
            sub_problems=sub_desc,
            approach="（在此描述总体建模思路，建议配流程图）",
            fig_ref="图1",
        )

    def _generate_assumptions(self) -> str:
        if not self.section_gen.assumptions:
            self.section_gen.add_default_assumptions()
        return self.section_gen.generate_assumptions()

    def _generate_symbols(self) -> str:
        if not self.section_gen.symbols:
            # 添加默认符号
            self.section_gen.add_symbol("x_i", "决策变量", "")
            self.section_gen.add_symbol("Z", "目标函数值", "")
            self.section_gen.add_symbol("n", "样本数量", "")
        return self.section_gen.generate_symbols()

    def _generate_sub_problem_section(self, idx: int, problem: Dict) -> str:
        lines = []

        # 问题分析
        lines.append(f"\\subsection{{问题分析}}")
        lines.append(f"（在此分析问题{idx}的特点和求解思路）\n")

        # 模型建立
        lines.append(f"\\subsection{{模型建立}}")
        if problem.get("model_name"):
            lines.append(f"基于以上分析，本文建立了{problem['model_name']}。\n")
        lines.append("（在此描述模型的数学表达式，包括目标函数和约束条件）\n")
        lines.append("目标函数：")
        lines.append(r"\begin{equation}")
        lines.append(r"\min \quad Z = f(x_1, x_2, \ldots, x_n)")
        lines.append(r"\end{equation}")
        lines.append("")
        lines.append("约束条件：")
        lines.append(r"\begin{equation}")
        lines.append(r"\text{s.t.} \quad g_i(x_1, x_2, \ldots, x_n) \leq b_i, \quad i = 1, 2, \ldots, m")
        lines.append(r"\end{equation}")
        lines.append("")

        # 模型求解
        lines.append(f"\\subsection{{模型求解}}")
        lines.append(f"采用{problem['method']}对模型进行求解。\n")
        lines.append("（在此描述求解过程和算法步骤）\n")

        # 结果展示
        lines.append(f"\\subsection{{结果与分析}}")
        if problem.get("result"):
            lines.append(f"求解得到：{problem['result']}。\n")
        else:
            lines.append("（在此展示求解结果，配图表说明）\n")
        lines.append("如表\\ref{tab:result" + str(idx) + "}所示。")
        lines.append("")
        lines.append(r"\begin{table}[H]")
        lines.append(r"  \centering")
        lines.append(r"  \caption{问题" + str(idx) + r"求解结果}")
        lines.append(r"  \label{tab:result" + str(idx) + r"}")
        lines.append(r"  \begin{tabular}{ccc}")
        lines.append(r"    \toprule")
        lines.append(r"    方案 & 指标1 & 指标2 \\")
        lines.append(r"    \midrule")
        lines.append(r"    方案A & — & — \\")
        lines.append(r"    方案B & — & — \\")
        lines.append(r"    \bottomrule")
        lines.append(r"  \end{tabular}")
        lines.append(r"\end{table}")

        return "\n".join(lines)

    def _generate_sensitivity_section(self) -> str:
        lines = [
            "为了检验模型的稳健性，我们对关键参数进行灵敏度分析。",
            "",
            "选取以下参数进行分析：",
            "",
        ]

        for i, p in enumerate(self.sensitivity_params, 1):
            lines.append(f"（{i}）{p['name']}（基准值：{p['base_value']}）")

        lines.append("")
        lines.append(
            "将各参数在其基准值附近分别变化±10\\%、±20\\%、±30\\%，"
            "观察目标函数的变化情况。结果如图\\ref{fig:sensitivity}所示。"
        )
        lines.append("")
        lines.append("（在此插入灵敏度分析图表）")
        lines.append("")
        lines.append(
            "由灵敏度分析结果可知，模型对关键参数的变化具有较好的稳健性，"
            "说明模型结果是可靠的。"
        )
        return "\n".join(lines)

    def _generate_evaluation_section(self) -> str:
        if not self.evaluators:
            # 生成默认评价
            lines = [
                "\\subsection{模型优点}",
                "（在此列出模型的优点，2-4条）\n",
                "\\subsection{模型不足}",
                "（在此诚实地指出模型的局限性）\n",
                "\\subsection{改进方向}",
                "（在此提出可能的改进方向）\n",
                "\\subsection{推广应用}",
                "（在此讨论模型的推广应用前景）",
            ]
            return "\n".join(lines)

        content = ""
        for ev in self.evaluators:
            content += ev.generate()
        return content

    def add_evaluator(self, evaluator):
        """添加模型评价器."""
        self.evaluators.append(evaluator)
        return self
