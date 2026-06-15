"""
摘要自动生成器

根据模型分析结果自动生成符合数模论文规范的摘要。

Example:
    >>> from mathflow.paper import AbstractGenerator
    >>> ag = AbstractGenerator("农作物种植策略优化")
    >>> ag.add_problem_solution(
    ...     description="建立最优种植方案",
    ...     method="线性规划 + 遗传算法",
    ...     result="最优年收益 285.6 万元",
    ... )
    >>> print(ag.generate())
"""

from typing import List, Dict, Optional


class AbstractGenerator:
    """
    数模论文摘要生成器.

    基于优秀论文的"五段式"摘要结构:
    1. 问题背景 (1句)
    2. 问题一的模型和结果
    3. 问题二的模型和结果
    4. 模型优势/灵敏度分析
    5. 关键词
    """

    def __init__(self, problem_title: str, background: str = ""):
        self.problem_title = problem_title
        self.background = background
        self.solutions = []
        self.model_strengths = []
        self.keywords = []

    def add_problem_solution(self, description: str, method: str,
                             result: str, model_name: str = ""):
        """
        添加一个问题的求解方案.

        Parameters
        ----------
        description : str
            问题描述
        method : str
            使用的方法/算法
        result : str
            关键结果 (数值)
        model_name : str
            模型名称
        """
        self.solutions.append({
            "description": description,
            "method": method,
            "result": result,
            "model_name": model_name,
        })
        return self

    def add_strength(self, strength: str):
        """添加模型优点."""
        self.model_strengths.append(strength)
        return self

    def set_keywords(self, keywords: List[str]):
        """设置关键词."""
        self.keywords = keywords
        return self

    def generate(self, style: str = "standard") -> str:
        """
        生成摘要.

        Parameters
        ----------
        style : str
            "standard" (标准五段式), "concise" (精简版), "detailed" (详细版)
        """
        if style == "standard":
            return self._generate_standard()
        elif style == "concise":
            return self._generate_concise()
        elif style == "detailed":
            return self._generate_detailed()
        else:
            raise ValueError(f"未知风格: {style}")

    def _generate_standard(self) -> str:
        """标准五段式摘要."""
        lines = []

        # 第一段: 背景 + 问题概述
        if self.background:
            lines.append(f"{self.background}")
        else:
            lines.append(f"本文针对{self.problem_title}问题进行了深入研究。")

        # 中间段: 各问题的求解
        for i, sol in enumerate(self.solutions, 1):
            model_desc = f"建立了{sol['model_name']}模型，" if sol["model_name"] else ""
            lines.append(
                f"对于问题{i}（{sol['description']}），"
                f"本文{model_desc}"
                f"采用{sol['method']}进行求解，"
                f"得到{sol['result']}。"
            )

        # 倒数第二段: 模型优势
        if self.model_strengths:
            strengths_text = "；".join(self.model_strengths)
            lines.append(f"本文所建模型具有以下特点：{strengths_text}。")

        # 最后一段: 关键词
        if self.keywords:
            kw_text = "；".join(self.keywords[:5])
            lines.append(f"\n关键词：{kw_text}")

        return "\n\n".join(lines)

    def _generate_concise(self) -> str:
        """精简版摘要 (300字以内)."""
        lines = []
        lines.append(f"本文针对{self.problem_title}问题，")

        methods = [sol["method"] for sol in self.solutions]
        lines.append(f"综合运用了{'、'.join(methods)}等方法进行分析。")

        for i, sol in enumerate(self.solutions, 1):
            lines.append(f"问题{i}：采用{sol['method']}，{sol['result']}。")

        if self.keywords:
            lines.append(f"\n关键词：{'；'.join(self.keywords[:5])}")

        return "\n".join(lines)

    def _generate_detailed(self) -> str:
        """详细版摘要 (500字左右)."""
        lines = []

        # 背景
        if self.background:
            lines.append(self.background)
        else:
            lines.append(f"本文针对{self.problem_title}问题，进行了系统深入的研究。")

        lines.append("")  # 空行

        # 各问题详细描述
        for i, sol in enumerate(self.solutions, 1):
            model_desc = f"建立了{sol['model_name']}模型，" if sol["model_name"] else ""
            lines.append(
                f"针对问题{i}（{sol['description']}），"
                f"{model_desc}"
                f"采用{sol['method']}进行求解。"
                f"{sol['result']}。"
            )
            lines.append("")

        # 模型优势
        if self.model_strengths:
            lines.append("本文模型的主要优势包括：")
            for j, s in enumerate(self.model_strengths, 1):
                lines.append(f"（{j}）{s}；")
            lines.append("")

        # 结论
        lines.append("本文所建模型具有良好的实用价值和推广性，可为相关决策提供科学依据。")

        # 关键词
        if self.keywords:
            lines.append(f"\n关键词：{'；'.join(self.keywords[:5])}")

        return "\n".join(lines)

    def word_count(self) -> int:
        """统计摘要字数."""
        return len(self.generate("concise").replace("\n", "").replace(" ", ""))
