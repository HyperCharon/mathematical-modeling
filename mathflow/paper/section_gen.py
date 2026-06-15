"""
论文章节自动生成器

自动生成数模论文的标准章节: 模型假设、符号说明、问题分析等。

Example:
    >>> from mathflow.paper import SectionGenerator
    >>> sg = SectionGenerator()
    >>> sg.add_assumption("假设数据真实可靠")
    >>> sg.add_symbol("x", "决策变量", "万元")
    >>> print(sg.generate_assumptions())
    >>> print(sg.generate_symbols())
"""

from typing import List, Dict, Optional


class SectionGenerator:
    """论文章节生成器."""

    def __init__(self):
        self.assumptions = []
        self.symbols = []
        self.references = []
        self.data_description = ""

    # ========== 模型假设 ==========

    def add_assumption(self, text: str, category: str = "general"):
        """
        添加模型假设.

        Parameters
        ----------
        text : str
            假设内容
        category : str
            类别: "data", "simplification", "continuity", "boundary", "general"
        """
        self.assumptions.append({"text": text, "category": category})
        return self

    def add_default_assumptions(self):
        """添加常用默认假设."""
        defaults = [
            ("假设题目中所给数据真实可靠，能够反映实际情况", "data"),
            ("假设忽略次要因素的影响，仅考虑主要因素", "simplification"),
            ("假设系统处于稳定状态", "continuity"),
            ("假设各因素之间相互独立", "simplification"),
            ("假设不考虑随机扰动的影响", "simplification"),
        ]
        for text, cat in defaults:
            self.add_assumption(text, cat)
        return self

    def generate_assumptions(self, title: str = "模型假设") -> str:
        """生成模型假设章节."""
        lines = [f"### {title}\n"]
        lines.append("为了简化问题，便于建立数学模型，本文作出以下假设：\n")
        for i, a in enumerate(self.assumptions, 1):
            lines.append(f"**假设{i}**：{a['text']}。")
        lines.append("")
        return "\n".join(lines)

    # ========== 符号说明 ==========

    def add_symbol(self, symbol: str, meaning: str, unit: str = "", category: str = ""):
        """
        添加符号说明.

        Parameters
        ----------
        symbol : str
            符号 (LaTeX 格式)
        meaning : str
            含义
        unit : str
            单位
        category : str
            类别: "决策变量", "状态变量", "参数", "集合" 等
        """
        self.symbols.append({
            "symbol": symbol, "meaning": meaning,
            "unit": unit, "category": category,
        })
        return self

    def generate_symbols(self, title: str = "符号说明") -> str:
        """生成符号说明章节 (三线表格式)."""
        lines = [f"### {title}\n"]
        lines.append("为便于描述，本文定义以下符号：\n")
        lines.append("| 符号 | 含义 | 单位 |")
        lines.append("|:---:|:---|:---:|")

        for s in self.symbols:
            unit = s["unit"] if s["unit"] else "—"
            lines.append(f"| ${s['symbol']}$ | {s['meaning']} | {unit} |")

        lines.append("")
        return "\n".join(lines)

    # ========== 问题分析 ==========

    def generate_problem_analysis(self, problem: str, sub_problems: List[Dict],
                                   approach: str = "", fig_ref: str = "") -> str:
        """
        生成问题分析章节.

        Parameters
        ----------
        problem : str
            问题描述
        sub_problems : list of dict
            子问题列表, 每个 dict 包含 "description" 和 "approach"
        approach : str
            总体建模思路
        fig_ref : str
            流程图引用
        """
        lines = ["### 问题分析\n"]
        lines.append(f"{problem}\n")

        if approach:
            lines.append(f"{approach}\n")

        if fig_ref:
            lines.append(f"本文的总体建模思路如{fig_ref}所示。\n")

        lines.append("通过分析，我们将原问题分解为以下子问题：\n")
        for i, sp in enumerate(sub_problems, 1):
            lines.append(f"**问题{i}**：{sp['description']}")
            if "approach" in sp:
                lines.append(f"拟采用{sp['approach']}进行求解。")
            lines.append("")

        return "\n".join(lines)

    # ========== 参考文献 ==========

    def add_reference(self, ref_type: str, **kwargs):
        """
        添加参考文献.

        Parameters
        ----------
        ref_type : str
            类型: "journal", "book", "conference", "online"
        """
        self.references.append({"type": ref_type, **kwargs})
        return self

    def add_common_references(self):
        """添加常用参考文献."""
        common = [
            {"type": "book", "num": 1, "authors": "司守奎, 孙兆亮",
             "title": "数学建模算法与应用", "city": "北京",
             "publisher": "国防工业出版社", "year": "2020"},
            {"type": "book", "num": 2, "authors": "姜启源, 谢金星, 叶俊",
             "title": "数学模型", "city": "北京",
             "publisher": "高等教育出版社", "year": "2018"},
            {"type": "book", "num": 3, "authors": "卓金武, 李必文",
             "title": "MATLAB在数学建模中的应用", "city": "北京",
             "publisher": "北京航空航天大学出版社", "year": "2020"},
            {"type": "book", "num": 4, "authors": "薛定宇, 陈阳泉",
             "title": "高等应用数学问题的MATLAB求解", "city": "北京",
             "publisher": "清华大学出版社", "year": "2018"},
        ]
        for ref in common:
            self.references.append(ref)
        return self

    def generate_references(self, title: str = "参考文献") -> str:
        """生成参考文献章节."""
        lines = [f"### {title}\n"]
        for ref in self.references:
            if ref["type"] == "journal":
                lines.append(
                    f"[{ref.get('num', '')}] {ref['authors']}. "
                    f"{ref['title']}[J]. {ref['journal']}, "
                    f"{ref['year']}, {ref.get('volume', '')}({ref.get('issue', '')}): "
                    f"{ref.get('pages', '')}."
                )
            elif ref["type"] == "book":
                lines.append(
                    f"[{ref.get('num', '')}] {ref['authors']}. "
                    f"{ref['title']}[M]. {ref['city']}: {ref['publisher']}, {ref['year']}."
                )
            elif ref["type"] == "online":
                lines.append(
                    f"[{ref.get('num', '')}] {ref['authors']}. "
                    f"{ref['title']}[EB/OL]. {ref.get('url', '')}."
                )
        lines.append("")
        return "\n".join(lines)

    # ========== 数据描述 ==========

    def set_data_description(self, description: str):
        """设置数据描述."""
        self.data_description = description
        return self

    def generate_data_description(self, title: str = "数据预处理") -> str:
        """生成数据描述/预处理章节."""
        lines = [f"### {title}\n"]
        if self.data_description:
            lines.append(self.data_description)
        else:
            lines.append(
                "本文对题目所给数据进行了以下预处理：\n"
                "1. 缺失值处理：采用均值/中位数填充；\n"
                "2. 异常值检测：采用IQR方法识别并处理异常值；\n"
                "3. 数据标准化：对不同量纲的指标进行归一化处理。"
            )
        lines.append("")
        return "\n".join(lines)
