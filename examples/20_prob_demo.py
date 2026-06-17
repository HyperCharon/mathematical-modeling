"""
示例 20: 概率分布拟合 + 假设检验

演示:
1. DistributionFitter — 自动拟合 8 种常见分布
2. HypothesisTest — t 检验、正态性检验、方差齐性检验
"""

import numpy as np
from mathflow.prob import DistributionFitter, HypothesisTest

np.random.seed(42)

print("=" * 60)
print("  1. 概率分布拟合 (DistributionFitter)")
print("=" * 60)

# 生成模拟数据: 产品寿命 (服从威布尔分布)
data = np.random.weibull(2, 300) * 100 + 50
print(f"\n数据: 300 个产品寿命样本, 范围 [{data.min():.1f}, {data.max():.1f}]")

fitter = DistributionFitter(data)
print(fitter)

# 自动拟合所有分布
results = fitter.auto_fit()
print(f"\n拟合了 {len(results)} 种分布:")
print(f"{'分布':>12s}  {'K-S统计量':>10s}  {'p值':>10s}  {'AIC':>10s}")
print("-" * 50)
for r in results:
    print(f"{r.distribution:>12s}  {r.ks_statistic:>10.4f}  {r.ks_p_value:>10.4f}  {r.aic:>10.1f}")

best = results[0]
print(f"\n最佳拟合: {best.distribution}")
print(f"  K-S 检验 p值 = {best.ks_p_value:.4f} {'✅ 不拒绝原分布' if best.ks_p_value > 0.05 else '❌ 拒绝原分布'}")

# 绘制拟合图
try:
    fig = fitter.plot()
    print("  [图] 分布拟合图已生成")
except Exception:
    pass

print("\n" + "=" * 60)
print("  2. 假设检验 (HypothesisTest)")
print("=" * 60)

ht = HypothesisTest(alpha=0.05)
print(f"\n{ht}")

# 2a. 单样本 t 检验
print("\n--- 单样本 t 检验 ---")
sample = np.random.normal(100, 15, 50)
result = ht.one_sample_ttest(sample, mu0=100)
print(f"  H0: μ = 100")
print(f"  t = {result.statistic:.4f}, p = {result.p_value:.4f}")
print(f"  结论: {result.conclusion}")

# 2b. 两样本 t 检验
print("\n--- 两样本 t 检验 ---")
group_a = np.random.normal(75, 10, 40)
group_b = np.random.normal(80, 12, 40)
result = ht.two_sample_ttest(group_a, group_b)
print(f"  H0: μ_A = μ_B")
print(f"  t = {result.statistic:.4f}, p = {result.p_value:.4f}")
print(f"  结论: {result.conclusion}")

# 2c. 正态性检验 (Shapiro-Wilk)
print("\n--- 正态性检验 (Shapiro-Wilk) ---")
normal_data = np.random.normal(0, 1, 100)
result = ht.normality_test(normal_data)
print(f"  统计量 = {result.statistic:.4f}, p = {result.p_value:.4f}")
print(f"  结论: {result.conclusion}")

# 2d. 方差齐性检验 (Levene)
print("\n--- 方差齐性检验 (Levene) ---")
g1 = np.random.normal(0, 1, 50)
g2 = np.random.normal(0, 2, 50)
result = ht.levene_test(g1, g2)
print(f"  统计量 = {result.statistic:.4f}, p = {result.p_value:.4f}")
print(f"  结论: {result.conclusion}")

print("\n✅ 概率统计示例完成!")
