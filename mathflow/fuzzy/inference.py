"""
模糊推理系统 (Mamdani 型)

Example:
    >>> from mathflow.fuzzy import FuzzyInference
    >>> fi = FuzzyInference()
    >>> fi.add_input("温度", range_=(0, 40), [("冷", "gaussian", 10, 5), ("适中", "gaussian", 20, 5), ("热", "gaussian", 30, 5)])
    >>> fi.add_output("风速", range_=(0, 100), [("低", "triangle", 0, 0, 50), ("高", "triangle", 50, 100, 100)])
    >>> fi.add_rule({"温度": "冷"}, "风速", "高")
"""

import numpy as np
from typing import Dict, List, Tuple
from mathflow.fuzzy.membership import MembershipFunction


class FuzzyInference:
    """
    Mamdani 型模糊推理系统.

    支持:
    - 多输入多输出
    - 多种隶属函数
    - 多种推理方法 (min, prod)
    - 多种解模糊方法 (重心法, 最大隶属度法)
    """

    def __init__(self):
        self.inputs = {}   # {name: (range_, terms)}
        self.outputs = {}
        self.rules = []
        self.mf = MembershipFunction()

    def __repr__(self) -> str:
        return f"FuzzyInference(inputs={len(self.inputs)}, outputs={len(self.outputs)}, rules={len(self.rules)})"

    def add_input(self, name: str, range_: Tuple, terms: List):
        """
        添加输入变量.

        Parameters
        ----------
        name : str
            变量名
        range_ : tuple (min, max)
            取值范围
        terms : list of (term_name, mf_type, *params)
            模糊集, 如 [("冷", "gaussian", 10, 5), ("热", "gaussian", 30, 5)]
        """
        self.inputs[name] = (range_, terms)
        return self

    def add_output(self, name: str, range_: Tuple, terms: List):
        """添加输出变量."""
        self.outputs[name] = (range_, terms)
        return self

    def add_rule(self, antecedents: Dict, consequent_var: str, consequent_term: str,
                 operator: str = "and"):
        """
        添加模糊规则.

        Parameters
        ----------
        antecedents : dict
            前件, 如 {"温度": "冷", "湿度": "高"}
        consequent_var : str
            后件变量名
        consequent_term : str
            后件模糊集名
        operator : str
            "and" (取小) 或 "or" (取大)
        """
        self.rules.append({
            "antecedents": antecedents,
            "consequent_var": consequent_var,
            "consequent_term": consequent_term,
            "operator": operator,
        })
        return self

    def _fuzzify(self, var_name, value, var_type="input"):
        """模糊化."""
        if var_type == "input":
            range_, terms = self.inputs[var_name]
        else:
            range_, terms = self.outputs[var_name]

        result = {}
        for term in terms:
            name = term[0]
            mf_type = term[1]
            params = term[2:]

            if mf_type == "triangle":
                result[name] = self.mf.triangle(value, *params)
            elif mf_type == "trapezoid":
                result[name] = self.mf.trapezoid(value, *params)
            elif mf_type == "gaussian":
                result[name] = self.mf.gaussian(value, *params)
            elif mf_type == "sigmoid":
                result[name] = self.mf.sigmoid(value, *params)
            else:
                raise ValueError(f"未知隶属函数: {mf_type}")

        return result

    def infer(self, input_values: Dict, method: str = "min",
              defuzzify: str = "centroid") -> Dict:
        """
        模糊推理.

        Parameters
        ----------
        input_values : dict
            输入值, 如 {"温度": 15, "湿度": 80}
        method : str
            推理方法: "min" (取小), "prod" (乘积)
        defuzzify : str
            解模糊方法: "centroid" (重心法), "max" (最大隶属度)
        """
        # Step 1: 模糊化
        fuzzified = {}
        for var, value in input_values.items():
            if var in self.inputs:
                fuzzified[var] = self._fuzzify(var, value, "input")

        # Step 2: 规则推理
        output_activation = {}
        for rule in self.rules:
            # 计算前件激活度
            activations = []
            for var, term in rule["antecedents"].items():
                if var in fuzzified and term in fuzzified[var]:
                    activations.append(fuzzified[var][term])

            if not activations:
                continue

            if rule["operator"] == "and":
                fire_strength = min(activations)
            else:
                fire_strength = max(activations)

            # 应用到后件
            out_var = rule["consequent_var"]
            out_term = rule["consequent_term"]

            if out_var not in output_activation:
                output_activation[out_var] = {}
            if out_term not in output_activation[out_var]:
                output_activation[out_var][out_term] = 0
            output_activation[out_var][out_term] = max(
                output_activation[out_var][out_term], fire_strength
            )

        # Step 3: 解模糊
        results = {}
        for out_var, term_strengths in output_activation.items():
            if out_var not in self.outputs:
                continue

            range_, terms = self.outputs[out_var]
            x = np.linspace(range_[0], range_[1], 200)

            # 聚合输出
            aggregated = np.zeros_like(x)
            for term in terms:
                name = term[0]
                if name in term_strengths:
                    strength = term_strengths[name]
                    mf_type = term[1]
                    params = term[2:]

                    if mf_type == "triangle":
                        mf_values = self.mf.triangle(x, *params)
                    elif mf_type == "trapezoid":
                        mf_values = self.mf.trapezoid(x, *params)
                    elif mf_type == "gaussian":
                        mf_values = self.mf.gaussian(x, *params)
                    else:
                        mf_values = np.zeros_like(x)

                    if method == "min":
                        clipped = np.minimum(mf_values, strength)
                    else:
                        clipped = mf_values * strength

                    aggregated = np.maximum(aggregated, clipped)

            # 解模糊
            if defuzzify == "centroid":
                if aggregated.sum() > 0:
                    results[out_var] = np.sum(x * aggregated) / np.sum(aggregated)
                else:
                    results[out_var] = (range_[0] + range_[1]) / 2
            elif defuzzify == "max":
                results[out_var] = x[np.argmax(aggregated)]

        return results

    def plot_rules(self, figsize=(12, 6)):
        """绘制规则激活情况."""
        import matplotlib.pyplot as plt

        n_rules = len(self.rules)
        fig, ax = plt.subplots(figsize=figsize)

        rule_labels = []
        for i, rule in enumerate(self.rules):
            antecedents = [f"{var}={term}" for var, term in rule["antecedents"].items()]
            consequent = f"{rule['consequent_var']}={rule['consequent_term']}"
            rule_labels.append(f"IF {' AND '.join(antecedents)} THEN {consequent}")

        ax.barh(range(n_rules), [1] * n_rules, color="lightblue", alpha=0.5)
        ax.set_yticks(range(n_rules))
        ax.set_yticklabels(rule_labels, fontsize=9)
        ax.set_xlabel("规则")
        ax.set_title("模糊规则库")
        ax.invert_yaxis()
        plt.tight_layout()
        return fig
