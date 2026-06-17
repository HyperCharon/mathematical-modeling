"""
示例 22: 综合评价方法大全

演示 6 种评价方法:
1. CRITIC — 客观赋权
2. 灰色关联分析 (GRA)
3. 秩和比法 (RSR)
4. PROMETHEE — 偏好排序
5. DEA — 数据包络分析
6. 模糊综合评价

场景: 6 家医院的护理质量综合评价
"""

import numpy as np
from mathflow.evaluate import (
    CRITIC, GreyRelationalAnalysis, RSR, PROMETHEE,
    DEA, FuzzyEvaluation
)

np.random.seed(42)

# ===== 评价数据 =====
# 6 家医院, 5 个指标: 病床使用率(%), 平均住院日, 治愈好转率(%), 院内感染率(%), 护理差错率(‰)
data = np.array([
    [95.2, 8.5, 92.1, 3.2, 0.8],
    [88.7, 10.2, 89.5, 4.1, 1.2],
    [92.1, 9.0, 94.3, 2.8, 0.5],
    [85.3, 11.5, 87.2, 5.0, 1.5],
    [90.5, 9.8, 91.0, 3.5, 0.9],
    [93.8, 8.0, 95.5, 2.5, 0.3],
])
labels = ["医院A", "医院B", "医院C", "医院D", "医院E", "医院F"]
# 指标类型: 1=效益型(越大越好), -1=成本型(越小越好)
types = [1, -1, 1, -1, -1]

print("=" * 60)
print("  综合评价: 6 家医院护理质量")
print("=" * 60)
print(f"样本: {len(labels)} 家医院, {data.shape[1]} 个指标")
print(f"指标类型: {['效益', '成本', '效益', '成本', '成本']}")

# ===== 1. CRITIC 客观赋权 =====
print("\n--- 1. CRITIC 客观赋权 ---")
critic = CRITIC(data, types=types)
critic.fit()
print(critic.summary(labels=labels))

# ===== 2. 灰色关联分析 =====
print("\n--- 2. 灰色关联分析 (GRA) ---")
# 参考序列: 各指标最优值
ref = np.array([data[:, 0].max(), data[:, 1].min(),
                data[:, 2].max(), data[:, 3].min(), data[:, 4].min()])
gra = GreyRelationalAnalysis(ref, data, rho=0.5)
gra.fit()
print(gra.summary(labels=labels))

# ===== 3. 秩和比法 (RSR) =====
print("\n--- 3. 秩和比法 (RSR) ---")
rsr = RSR(data, types=types)
rsr.fit()
print(rsr.summary(labels=labels))

# ===== 4. PROMETHEE 偏好排序 =====
print("\n--- 4. PROMETHEE 偏好排序 ---")
prom = PROMETHEE(data, types=types, weights=[0.25, 0.15, 0.30, 0.15, 0.15])
prom.fit()
print(prom.summary())

# ===== 5. DEA 数据包络分析 =====
print("\n--- 5. DEA 数据包络分析 (CCR 模型) ---")
# 输入: 病床使用率, 平均住院日 (投入指标)
# 输出: 治愈好转率 (产出指标)
dea = DEA(inputs=data[:, :2], outputs=data[:, 2:3], model="CCR")
dea.fit()
print(dea.summary())

# ===== 6. 模糊综合评价 =====
print("\n--- 6. 模糊综合评价 ---")
# 评价等级
grades = ["优秀", "良好", "一般", "较差"]

# 构建模糊评价矩阵 (5个指标 × 4个等级)
R = np.array([
    [0.6, 0.3, 0.1, 0.0],  # 病床使用率
    [0.2, 0.4, 0.3, 0.1],  # 平均住院日
    [0.5, 0.3, 0.2, 0.0],  # 治愈好转率
    [0.3, 0.4, 0.2, 0.1],  # 院内感染率
    [0.7, 0.2, 0.1, 0.0],  # 护理差错率
])
indicator_weights = [0.25, 0.15, 0.30, 0.15, 0.15]

fuzzy = FuzzyEvaluation(indicator_weights, R, grade_names=grades)
fuzzy.fit()
print(fuzzy.summary())

# ===== 综合排名汇总 =====
print("\n" + "=" * 60)
print("  综合排名汇总")
print("=" * 60)

# 获取各方法排名
print(f"\n{'医院':>6s}  {'GRA排名':>8s}  {'RSR排名':>8s}  {'PROMETHEE':>10s}")
print("-" * 40)
for i, label in enumerate(labels):
    gra_r = int(np.where(np.argsort(-gra.correlations) == i)[0][0]) + 1
    rsr_r = int(rsr.rankings[i])
    prom_r = int(prom.rankings[i])
    print(f"{label:>6s}  {gra_r:>8d}  {rsr_r:>8d}  {prom_r:>10d}")

print("\n✅ 综合评价示例完成!")
