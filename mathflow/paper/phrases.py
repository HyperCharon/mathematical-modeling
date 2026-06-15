"""
数模论文常用语料库

基于优秀论文总结的高频句式和过渡词，写作时直接调用。
"""


class Phrases:
    """数模论文常用语料库."""

    # ========== 摘要常用句式 ==========
    abstract = {
        "opening": [
            "本文针对{problem}问题，综合运用了{methods}等方法，对{content}进行了深入分析。",
            "本文围绕{problem}展开研究，通过建立{model}模型，解决了{content}问题。",
            "针对{problem}这一实际问题，本文基于{theory}理论，构建了{model}模型。",
            "本文以{problem}为研究对象，采用{method}与{method2}相结合的方法，对{content}进行了系统研究。",
        ],
        "problem_intro": [
            "对于问题一（{description}），本文建立了{model}模型。",
            "针对第一个子问题，考虑到{factor}等因素，本文构建了{model}。",
            "在问题一的求解中，本文首先对{data}进行了预处理，随后建立了{model}。",
        ],
        "method_result": [
            "采用{algorithm}算法求解，得到{result}。",
            "利用{tool}编程求解，结果表明{conclusion}。",
            "通过对模型进行数值模拟，得到{result}，误差控制在{error}以内。",
            "运用{method}对模型进行求解，经过{n}次迭代后收敛，最优解为{result}。",
        ],
        "prediction": [
            "基于该模型，对未来{period}的{target}进行了预测，预测值为{value}。",
            "利用训练好的模型进行预测，{target}的预测精度达到{accuracy}。",
        ],
        "evaluation": [
            "模型的拟合优度R²={r2}，表明模型具有较好的解释能力。",
            "灵敏度分析表明，模型对关键参数的变化具有较好的稳健性。",
            "与传统方法相比，本文模型的{metric}提升了{percent}%。",
        ],
        "closing": [
            "本文所建模型具有良好的推广性和实用价值，可为{field}提供科学的决策依据。",
            "本文的方法不仅适用于{problem}，还可以推广应用于{other_field}领域。",
            "本文的研究成果对于{field}具有一定的理论意义和实际应用价值。",
        ],
        "keywords": "关键词：{kw1}；{kw2}；{kw3}；{kw4}；{kw5}",
    }

    # ========== 正文过渡词 ==========
    transitions = {
        "addition": ["此外", "另外", "不仅如此", "更重要的是", "进一步而言", "在此基础上"],
        "contrast": ["然而", "但是", "尽管如此", "另一方面", "反之", "与此不同的是"],
        "cause": ["因此", "由此可知", "鉴于此", "基于上述分析", "所以", "这说明"],
        "summary": ["综上所述", "总而言之", "归纳起来", "概括地说", "综合以上分析"],
        "sequence": ["首先", "其次", "然后", "最后", "在此基础上", "紧接着"],
        "comparison": ["相比之下", "与...不同的是", "相较而言", "对比分析表明"],
        "condition": ["当...时", "若...则", "在...条件下", "假设...", "考虑到..."],
        "clarify": ["换言之", "即", "也就是说", "具体来说", "更准确地说"],
    }

    # ========== 模型描述常用句式 ==========
    model_description = {
        "abstract": [
            "通过分析可知，该问题可以抽象为一个{type}问题。",
            "该问题的本质是在{constraints}约束下，求解{objective}的最优解。",
            "考虑到{factor}的影响，我们将该问题转化为一个{type}模型。",
        ],
        "variables": [
            "我们引入决策变量$x_{i}$表示{meaning}。",
            "设{symbol}为{meaning}，其中{range}。",
            "定义状态变量$${symbol}(t)$$表示t时刻的{meaning}。",
        ],
        "objective": [
            "目标函数为：$${sense} \\quad Z = {expr}$$",
            "以{target}最小（大）化为优化目标，建立目标函数如式({eq_num})所示。",
        ],
        "constraints": [
            "约束条件如下：",
            "考虑以下约束：",
            "模型需满足以下条件：",
        ],
        "solution": [
            "利用{tool}对模型进行编程求解。",
            "采用{algorithm}算法，经过{n}次迭代后得到最优解。",
            "通过对模型进行数值模拟，得到如下结果。",
            "运用{method}对模型进行求解，收敛精度为{precision}。",
        ],
    }

    # ========== 结果分析常用句式 ==========
    results = {
        "figure": [
            "由图{num}可知，{description}。",
            "如图{num}所示，{description}，这与{theory}理论分析一致。",
            "图{num}直观地展示了{description}。",
        ],
        "table": [
            "从表{num}中可以看出，{description}。",
            "表{num}给出了{description}的结果。",
            "如表{num}所示，{key_finding}。",
        ],
        "data_analysis": [
            "结果表明，{model}能够较好地拟合实际数据，拟合优度R²={r2}。",
            "模型的预测误差为{error}，在可接受范围内。",
            "与{baseline}相比，本文方法的{metric}提高了{percent}%。",
            "分析发现，{variable}对{target}的影响最为显著，其灵敏度系数为{coef}。",
        ],
        "comparison": [
            "为验证模型的有效性，我们将本文方法与{method2}进行了对比。",
            "表{num}列出了不同方法的性能对比，可以看出本文方法在{metric}方面具有明显优势。",
            "对比分析表明，{method1}在{condition}下表现更优，而{method2}适用于{condition2}。",
        ],
    }

    # ========== 灵敏度分析常用句式 ==========
    sensitivity = {
        "intro": [
            "为了检验模型的稳健性，我们对关键参数进行了灵敏度分析。",
            "为评估参数变化对模型结果的影响程度，选取{n}个关键参数进行灵敏度分析。",
        ],
        "method": [
            "将参数{param}在其基准值{base}附近分别变化±10%、±20%、±30%，观察{target}的变化。",
            "采用单因素灵敏度分析方法，每次只改变一个参数，保持其他参数不变。",
            "运用Morris筛选法对{n}个参数进行全局灵敏度分析。",
        ],
        "result": [
            "如图{num}所示，当{param}变化{percent}%时，{target}仅变化了{result_percent}%，说明模型对该参数不敏感。",
            "灵敏度分析结果表明，{param}对模型结果的影响最大，而{param2}的影响较小。",
            "龙卷风图（图{num}）直观地展示了各参数的灵敏度排序。",
        ],
        "conclusion": [
            "综合以上分析，模型对关键参数的变化具有较好的稳健性，结果可靠。",
            "灵敏度分析验证了模型的鲁棒性，表明模型在参数波动范围内仍能给出稳定的结果。",
        ],
    }

    # ========== 模型评价常用句式 ==========
    evaluation = {
        "strengths": [
            "本文所建立的{model}具有以下优点：",
            "本文模型的主要优势在于：",
        ],
        "strength_items": [
            "模型综合考虑了{factors}等多种因素，具有较强的现实意义。",
            "采用{method}求解，计算效率高，收敛速度快。",
            "模型的拟合精度高，R²={r2}，预测效果好。",
            "模型具有良好的可扩展性，可以方便地引入新的约束条件。",
            "灵敏度分析表明模型具有较好的稳健性。",
        ],
        "weaknesses": [
            "不足之处在于：",
            "本文模型的局限性主要体现在：",
        ],
        "weakness_items": [
            "模型假设条件较为理想化，实际应用中可能需要进一步调整。",
            "受数据量限制，模型的泛化能力有待进一步验证。",
            "模型未考虑{factor}的影响，可能导致一定的偏差。",
        ],
        "improvement": [
            "若考虑{factor}因素，可以进一步改进模型为{improved_model}。",
            "未来可以引入{method}来提升模型的{aspect}。",
            "在后续研究中，可以将{factor}纳入模型，建立更加完善的{model}。",
        ],
        "extension": [
            "本文提出的{method}不仅适用于{problem}，还可以推广应用于{other_field}。",
            "该模型可以为{field}领域的类似问题提供参考和借鉴。",
        ],
    }

    # ========== 问题分析常用句式 ==========
    problem_analysis = {
        "background": [
            "{problem}是一个具有重要实际意义的问题，涉及到{fields}等多个领域。",
            "该问题的核心在于如何在{constraints}条件下，实现{objective}。",
        ],
        "approach": [
            "通过分析，我们将原问题分解为{n}个子问题：",
            "针对该问题的特点，我们拟采用以下建模思路：",
            "考虑到问题的复杂性，我们将其分为以下几个步骤进行求解：",
        ],
        "flowchart": [
            "图{num}给出了本文的总体建模思路和求解流程。",
            "如图{num}所示，本文的建模过程主要包括{steps}等步骤。",
        ],
    }

    # ========== 模型假设常用句式 ==========
    assumptions = {
        "data": [
            "假设题目中所给数据真实可靠，能够反映实际情况。",
            "假设数据采集过程中的误差可以忽略不计。",
        ],
        "simplification": [
            "假设忽略{factor}等次要因素的影响。",
            "假设系统处于{state}状态。",
            "假设各因素之间相互独立，不存在交互作用。",
        ],
        "continuity": [
            "假设{variable}的变化是连续的。",
            "假设模型参数在研究期间保持不变。",
        ],
        "boundary": [
            "假设研究范围仅限于{scope}。",
            "假设不考虑{factor}等极端情况。",
        ],
    }

    # ========== 参考文献格式 ==========
    references = {
        "journal": "[{num}] {authors}. {title}[J]. {journal}, {year}, {volume}({issue}): {pages}.",
        "book": "[{num}] {authors}. {title}[M]. {city}: {publisher}, {year}.",
        "conference": "[{num}] {authors}. {title}[C]//{conference}. {city}: {publisher}, {year}: {pages}.",
        "online": "[{num}] {authors}. {title}[EB/OL]. ({date})[{access_date}]. {url}.",
        "standard_format": "GB/T 7714-2015",
    }

    @classmethod
    def get(cls, category: str, key: str = None, **kwargs) -> str:
        """
        获取语料.

        Parameters
        ----------
        category : str
            类别: "abstract", "transitions", "model_description", "results",
            "sensitivity", "evaluation", "problem_analysis", "assumptions"
        key : str, optional
            具体条目名
        **kwargs
            模板参数
        """
        section = getattr(cls, category, None)
        if section is None:
            raise ValueError(f"未知类别: {category}")

        if key:
            items = section.get(key, [])
            if isinstance(items, list):
                import random
                template = random.choice(items)
            else:
                template = items
        else:
            # 返回整个类别的所有条目
            return section

        # 格式化模板
        for k, v in kwargs.items():
            template = template.replace(f"{{{k}}}", str(v))

        return template

    @classmethod
    def list_categories(cls):
        """列出所有类别."""
        return ["abstract", "transitions", "model_description", "results",
                "sensitivity", "evaluation", "problem_analysis", "assumptions"]
