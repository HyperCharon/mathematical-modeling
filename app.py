"""
MathFlow Web UI — Streamlit 仪表盘

运行方式:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="MathFlow 数学建模工具箱", page_icon="🧮", layout="wide")

st.title("🧮 MathFlow — 数学建模全流程工作台")

# 侧边栏
st.sidebar.title("功能导航")
page = st.sidebar.radio("选择模块", [
    "📊 综合评价 (AHP+TOPSIS)",
    "📈 预测模型",
    "⚙️ 优化求解",
    "🔬 ODE 求解",
    "📋 聚类分析",
    "🎯 模板一键运行",
])

if page == "📊 综合评价 (AHP+TOPSIS)":
    st.header("📊 AHP + TOPSIS 综合评价")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("AHP 判断矩阵")
        n = st.slider("指标数量", 2, 8, 3)
        matrix = np.ones((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                val = st.number_input(f"a[{i+1},{j+1}]", 1/9.0, 9.0, 3.0, 0.1, key=f"ahp_{i}_{j}")
                matrix[i][j] = val
                matrix[j][i] = 1.0 / val

        from mathflow.evaluate import AHP
        ahp = AHP()
        ahp.set_matrix(matrix)
        ahp.fit()

        st.metric("CR", f"{ahp.CR:.4f}", "✅ 通过" if ahp.is_consistent else "❌ 未通过")

    with col2:
        st.subheader("权重结果")
        fig = ahp.plot_weights()
        st.pyplot(fig)

    st.divider()
    st.subheader("TOPSIS 评价")

    n_options = st.number_input("方案数", 2, 20, 4)
    n_indicators = n

    data = np.zeros((n_options, n_indicators))
    cols = st.columns(min(n_indicators, 4))
    for j in range(n_indicators):
        with cols[j % len(cols)]:
            for i in range(n_options):
                data[i][j] = st.number_input(f"方案{i+1}-指标{j+1}", 0.0, 1000.0, 50.0, key=f"top_{i}_{j}")

    types = []
    for j in range(n_indicators):
        t = st.selectbox(f"指标{j+1}类型", ["效益型(越大越好)", "成本型(越小越好)"], key=f"type_{j}")
        types.append(1 if "效益" in t else -1)

    if st.button("运行 TOPSIS"):
        from mathflow.evaluate import TOPSIS
        topsis = TOPSIS(data, weights=ahp.weights, types=types)
        topsis.fit()
        st.dataframe({
            "方案": [f"方案{i+1}" for i in range(n_options)],
            "得分": topsis.scores,
            "排名": topsis.rankings,
        })

elif page == "📈 预测模型":
    st.header("📈 预测模型")

    method = st.selectbox("选择方法", ["灰色预测 GM(1,1)", "指数平滑", "曲线拟合"])

    data_input = st.text_area("输入数据 (逗号分隔)", "124, 130, 138, 146, 155, 165")
    data = np.array([float(x.strip()) for x in data_input.split(",")])
    steps = st.slider("预测步数", 1, 10, 3)

    if st.button("运行预测"):
        if "灰色" in method:
            from mathflow.predict import GreyPrediction
            model = GreyPrediction(data)
            model.fit()
            forecast = model.predict(steps)
            st.write(model.summary(steps))
            fig = model.plot(steps)
            st.pyplot(fig)

        elif "指数" in method:
            from mathflow.predict import ExponentialSmoothing
            model = ExponentialSmoothing(data, method="holt")
            model.fit()
            forecast = model.predict(steps)
            st.write(model.summary(steps))
            fig = model.plot(steps)
            st.pyplot(fig)

        elif "曲线" in method:
            from mathflow.interpolate import CurveFitter
            x = np.arange(1, len(data) + 1)
            model = CurveFitter(x, data)
            model.auto_fit()
            st.write(model.summary())
            fig = model.plot()
            st.pyplot(fig)

elif page == "⚙️ 优化求解":
    st.header("⚙️ 优化求解")

    n_vars = st.number_input("变量数", 1, 10, 2)
    sense = st.selectbox("目标", ["最大化", "最小化"])

    obj_coeffs = []
    for i in range(n_vars):
        obj_coeffs.append(st.number_input(f"目标系数 c{i+1}", -100.0, 100.0, 1.0, key=f"obj_{i}"))

    st.subheader("约束条件")
    n_constraints = st.number_input("约束数", 1, 10, 2)

    from mathflow.optimize import LinearProgramming
    lp = LinearProgramming()
    lp.set_objective(obj_coeffs, sense="max" if sense == "最大化" else "min")

    for k in range(int(n_constraints)):
        cols = st.columns(n_vars + 2)
        coeffs = []
        for i in range(n_vars):
            with cols[i]:
                coeffs.append(st.number_input(f"a{k+1},{i+1}", -100.0, 100.0, 1.0, key=f"c_{k}_{i}"))
        with cols[-2]:
            op = st.selectbox("", ["<=", ">=", "=="], key=f"op_{k}")
        with cols[-1]:
            rhs = st.number_input(f"b{k+1}", -1000.0, 1000.0, 10.0, key=f"rhs_{k}")
        lp.add_constraint(coeffs, op, rhs)

    if st.button("求解"):
        result = lp.solve()
        st.success(f"最优值: {result.optimal_value:.4f}")
        st.json(result.variables)

        if n_vars == 2:
            fig = lp.plot_feasible_region()
            st.pyplot(fig)

elif page == "🔬 ODE 求解":
    st.header("🔬 ODE 求解器")

    model_type = st.selectbox("模型", ["自定义 dy/dt = f(t,y)", "SIR 传染病", "阻尼振动"])

    if "SIR" in model_type:
        beta = st.slider("β (传染率)", 0.01, 1.0, 0.3)
        gamma = st.slider("γ (恢复率)", 0.01, 1.0, 0.1)
        I0 = st.slider("初始感染比例", 0.001, 0.5, 0.01)

        if st.button("模拟"):
            from mathflow.ode import ODESolver, sir_model
            solver = ODESolver(sir_model(beta=beta, gamma=gamma), y0=[1-I0, I0, 0])
            result = solver.solve(t_span=(0, 200), dt=0.5)
            st.write(f"**R0 = {beta/gamma:.2f}**")
            st.write(f"峰值感染: {result.y[:, 1].max():.2%} (第 {result.t[np.argmax(result.y[:, 1])]:.0f} 天)")
            st.write(f"最终感染: {result.y[-1, 2]:.2%}")
            fig = solver.plot(labels=["S 易感者", "I 感染者", "R 恢复者"])
            st.pyplot(fig)

elif page == "📋 聚类分析":
    st.header("📋 聚类分析")

    from sklearn.datasets import load_iris, load_wine
    dataset = st.selectbox("数据集", ["Iris 鸢尾花", "Wine 葡萄酒"])

    if "Iris" in dataset:
        data = load_iris()
    else:
        data = load_wine()

    X, y = data.data, data.target
    st.write(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}")

    method = st.selectbox("聚类方法", ["kmeans", "dbscan", "hierarchical"])
    k = st.slider("聚类数 K", 2, 10, 3)

    if st.button("运行聚类"):
        from mathflow.ml import ClusterAnalysis
        ca = ClusterAnalysis(X)
        ca.fit(method=method, n_clusters=k)
        st.write(ca.summary())
        fig = ca.plot_clusters()
        st.pyplot(fig)

elif page == "🎯 模板一键运行":
    st.header("🎯 赛题模板一键运行")

    from mathflow.templates import TemplateRunner, list_templates

    templates = list_templates()
    template = st.selectbox("选择模板", list(templates.keys()))

    st.info(f"**说明**: {templates[template]}")

    if st.button("🚀 运行模板"):
        with st.spinner("正在运行..."):
            tr = TemplateRunner(template)
            result = tr.run()
        st.success("运行完成！")
