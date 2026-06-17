"""
模型评价自动生成器

自动生成"模型评价与推广"章节，包括优缺点分析、改进方向。

Example:
    >>> from mathflow.paper import ModelEvaluator
    >>> me = ModelEvaluator("AHP-TOPSIS 综合评价模型")
    >>> me.add_strength("综合考虑了主观和客观权重")
    >>> me.add_strength("灵敏度分析验证了模型的稳健性")
    >>> me.add_weakness("AHP判断矩阵依赖专家经验")
    >>> me.add_improvement("引入博弈论组合赋权", "权重确定方法")
    >>> print(me.generate())
"""


class ModelEvaluator:
    """
    模型评价自动生成器.

    自动生成:
    - 优点分析
    - 缺点分析
    - 改进方向
    - 推广应用
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.strengths = []
        self.weaknesses = []
        self.improvements = []
        self.extensions = []
        self.metrics = {}  # 量化指标

    def add_strength(self, strength: str, evidence: str = ""):
        """添加优点."""
        self.strengths.append({"text": strength, "evidence": evidence})
        return self

    def add_weakness(self, weakness: str):
        """添加缺点."""
        self.weaknesses.append(weakness)
        return self

    def add_improvement(self, improvement: str, aspect: str = ""):
        """添加改进方向."""
        self.improvements.append({"text": improvement, "aspect": aspect})
        return self

    def add_extension(self, field: str, description: str = ""):
        """添加推广应用."""
        self.extensions.append({"field": field, "description": description})
        return self

    def set_metric(self, name: str, value: str):
        """设置量化指标."""
        self.metrics[name] = value
        return self

    def generate(self, style: str = "standard") -> str:
        """生成模型评价章节."""
        if style == "standard":
            return self._generate_standard()
        elif style == "brief":
            return self._generate_brief()
        else:
            raise ValueError(f"未知风格: {style}")

    def _generate_standard(self) -> str:
        """标准模型评价."""
        lines = []

        # 标题
        lines.append("## 模型评价与推广\n")

        # 优点
        if self.strengths:
            lines.append("### 模型优点\n")
            lines.append(f"本文所建立的{self.model_name}具有以下优点：\n")
            for i, s in enumerate(self.strengths, 1):
                text = f"（{i}）{s['text']}"
                if s["evidence"]:
                    text += f"。{s['evidence']}"
                text += "。"
                lines.append(text)
            lines.append("")

        # 缺点
        if self.weaknesses:
            lines.append("### 模型不足\n")
            lines.append(f"本文模型的局限性主要体现在以下几个方面：\n")
            for i, w in enumerate(self.weaknesses, 1):
                lines.append(f"（{i}）{w}。")
            lines.append("")

        # 改进方向
        if self.improvements:
            lines.append("### 改进方向\n")
            lines.append("针对上述不足，未来可以从以下方面进行改进：\n")
            for i, imp in enumerate(self.improvements, 1):
                aspect = f"在{imp['aspect']}方面，" if imp["aspect"] else ""
                lines.append(f"（{i}）{aspect}{imp['text']}。")
            lines.append("")

        # 推广应用
        if self.extensions:
            lines.append("### 推广应用\n")
            lines.append(f"本文提出的{self.model_name}不仅适用于当前问题，还可以推广应用于以下领域：\n")
            for ext in self.extensions:
                desc = f"：{ext['description']}" if ext["description"] else ""
                lines.append(f"- {ext['field']}{desc}")
            lines.append("")

        return "\n".join(lines)

    def _generate_brief(self) -> str:
        """精简版评价."""
        lines = []
        lines.append(f"### 模型评价\n")

        if self.strengths:
            strengths_text = "；".join([s["text"] for s in self.strengths])
            lines.append(f"**优点**：{strengths_text}。\n")

        if self.weaknesses:
            weaknesses_text = "；".join(self.weaknesses)
            lines.append(f"**不足**：{weaknesses_text}。\n")

        if self.improvements:
            imp_text = "；".join([i["text"] for i in self.improvements])
            lines.append(f"**改进方向**：{imp_text}。\n")

        return "\n".join(lines)

    def generate_tornado_description(self, param_names, sensitivities, fig_num=1):
        """
        生成龙卷风图的文字描述.

        Parameters
        ----------
        param_names : list
            参数名列表
        sensitivities : list
            灵敏度值列表
        fig_num : int
            图号
        """
        # 按灵敏度排序
        sorted_pairs = sorted(zip(param_names, sensitivities), key=lambda x: -x[1])
        most_sensitive = sorted_pairs[0]
        least_sensitive = sorted_pairs[-1]

        lines = [
            f"如图{fig_num}所示的龙卷风图直观地展示了各参数的灵敏度排序。",
            f"其中，{most_sensitive[0]}对模型结果的影响最大，灵敏度系数为{most_sensitive[1]:.4f}；",
            f"{least_sensitive[0]}的影响最小，灵敏度系数为{least_sensitive[1]:.4f}。",
            f"这说明模型对{most_sensitive[0]}较为敏感，在实际应用中需要重点关注该参数的取值。",
        ]
        return "\n".join(lines)

    def generate_sensitivity_conclusion(self, max_change: float, param: str, target: str):
        """
        生成灵敏度分析结论.

        Parameters
        ----------
        max_change : float
            最大变化百分比
        param : str
            参数名
        target : str
            目标名
        """
        if max_change < 5:
            level = "非常不敏感"
            robustness = "具有很强的稳健性"
        elif max_change < 10:
            level = "不敏感"
            robustness = "具有较好的稳健性"
        elif max_change < 20:
            level = "中等敏感"
            robustness = "稳健性一般"
        else:
            level = "敏感"
            robustness = "需要进一步优化以提高稳健性"

        return (
            f"当{param}变化30%时，{target}仅变化了{max_change:.1f}%，"
            f"说明模型对{param}{level}，{robustness}。"
        )
