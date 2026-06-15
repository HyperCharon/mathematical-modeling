"""论文写作模块测试."""

import pytest
from mathflow.paper import AbstractGenerator, ModelEvaluator, SectionGenerator, FullPaper, Phrases


class TestAbstractGenerator:
    def test_standard(self):
        ag = AbstractGenerator("测试问题")
        ag.add_problem_solution("问题描述", "AHP方法", "得分0.85", "评价模型")
        ag.set_keywords(["AHP", "评价"])
        text = ag.generate("standard")
        assert "测试问题" in text
        assert "AHP" in text
        assert "关键词" in text

    def test_concise(self):
        ag = AbstractGenerator("测试")
        ag.add_problem_solution("问题", "方法", "结果")
        text = ag.generate("concise")
        assert len(text) < 500

    def test_multiple_problems(self):
        ag = AbstractGenerator("测试")
        ag.add_problem_solution("问题1", "方法1", "结果1")
        ag.add_problem_solution("问题2", "方法2", "结果2")
        ag.add_problem_solution("问题3", "方法3", "结果3")
        text = ag.generate("standard")
        assert "问题一" in text or "问题1" in text
        assert "问题二" in text or "问题2" in text

    def test_word_count(self):
        ag = AbstractGenerator("测试问题描述")
        ag.add_problem_solution("问题", "方法", "结果")
        count = ag.word_count()
        assert count > 0


class TestModelEvaluator:
    def test_generation(self):
        me = ModelEvaluator("测试模型")
        me.add_strength("优点1")
        me.add_strength("优点2", "证据")
        me.add_weakness("缺点1")
        me.add_improvement("改进1")
        me.add_extension("应用领域")
        text = me.generate()
        assert "优点" in text
        assert "不足" in text
        assert "改进" in text
        assert "推广" in text

    def test_brief(self):
        me = ModelEvaluator("测试")
        me.add_strength("优点")
        text = me.generate("brief")
        assert "优点" in text

    def test_tornado_description(self):
        me = ModelEvaluator("测试")
        desc = me.generate_tornado_description(["A", "B"], [0.5, 0.3])
        assert "A" in desc

    def test_sensitivity_conclusion(self):
        me = ModelEvaluator("测试")
        desc = me.generate_sensitivity_conclusion(5.0, "参数A", "目标B")
        assert "参数A" in desc


class TestSectionGenerator:
    def test_assumptions(self):
        sg = SectionGenerator()
        sg.add_assumption("假设1")
        sg.add_assumption("假设2")
        text = sg.generate_assumptions()
        assert "假设1" in text
        assert "假设2" in text

    def test_default_assumptions(self):
        sg = SectionGenerator()
        sg.add_default_assumptions()
        text = sg.generate_assumptions()
        assert len(sg.assumptions) >= 3

    def test_symbols(self):
        sg = SectionGenerator()
        sg.add_symbol("x", "变量", "万元")
        sg.add_symbol("y", "因变量", "")
        text = sg.generate_symbols()
        assert "| 符号 |" in text
        assert "x" in text

    def test_references(self):
        sg = SectionGenerator()
        sg.add_common_references()
        text = sg.generate_references()
        assert "[1]" in text
        assert "司守奎" in text


class TestFullPaper:
    def test_generate(self):
        paper = FullPaper("测试论文")
        paper.set_background("背景描述")
        paper.add_sub_problem("问题1", "方法1", "结果1")
        paper.add_assumption("假设1")
        paper.add_symbol("x", "变量", "")
        paper.set_keywords(["关键词1", "关键词2"])
        content = paper.generate()
        assert "\\documentclass" in content
        assert "\\begin{abstract}" in content
        assert "\\section{问题重述}" in content
        assert "\\section{模型假设}" in content
        assert "\\section{符号说明}" in content
        assert "问题1的模型建立与求解" in content
        assert "\\end{document}" in content


class TestPhrases:
    def test_get_transitions(self):
        word = Phrases.get("transitions", "addition")
        assert isinstance(word, str)
        assert len(word) > 0

    def test_get_abstract(self):
        text = Phrases.get("abstract", "opening", problem="测试", methods="AHP", content="评价")
        assert "测试" in text

    def test_list_categories(self):
        cats = Phrases.list_categories()
        assert "abstract" in cats
        assert "transitions" in cats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
