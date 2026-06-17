"""
示例 21: 数据加载与清洗

演示:
1. DataLoader — CSV/Excel/JSON 加载
2. DataCleaner — 缺失值处理、异常值检测、数据标准化
"""

import numpy as np
import pandas as pd
from mathflow.data import DataLoader, DataCleaner

print("=" * 60)
print("  1. 数据加载 (DataLoader)")
print("=" * 60)

# 创建示例数据文件
sample_data = pd.DataFrame({
    "姓名": ["张三", "李四", "王五", "赵六", "钱七"],
    "年龄": [25, 30, 35, 28, 42],
    "收入": [8000, 12000, 15000, 9500, 20000],
    "学历": ["本科", "硕士", "博士", "本科", "硕士"],
})
sample_data.to_csv("/tmp/sample_data.csv", index=False, encoding="utf-8-sig")
sample_data.to_json("/tmp/sample_data.json", orient="records", force_ascii=False)

# 加载 CSV
df_csv = DataLoader.load_csv("/tmp/sample_data.csv")
print(f"\nCSV 加载: {df_csv.shape[0]} 行 × {df_csv.shape[1]} 列")
print(df_csv.to_string(index=False))

# 加载 JSON
df_json = DataLoader.load_json("/tmp/sample_data.json")
print(f"\nJSON 加载: {df_json.shape[0]} 行 × {df_json.shape[1]} 列")

print("\n" + "=" * 60)
print("  2. 数据清洗 (DataCleaner)")
print("=" * 60)

# 创建含缺失值和异常值的数据
np.random.seed(42)
dirty_data = pd.DataFrame({
    "产量": [100, 120, np.nan, 115, 500, 108, 125, np.nan, 110, 118],
    "温度": [25, 27, 26, np.nan, 28, -10, 27, 26, 25, 27],
    "湿度": [60, 65, 58, 62, 64, 61, np.nan, 59, 63, 100],
})
print(f"\n原始数据 ({dirty_data.shape[0]} 行):")
print(dirty_data.to_string(index=False))

# 2a. 缺失值处理
print("\n--- 缺失值处理 (均值填充) ---")
cleaned = DataCleaner.handle_missing(dirty_data, method="mean")
print(cleaned.round(2).to_string(index=False))

# 2b. 异常值检测
print("\n--- 异常值检测 (IQR 方法) ---")
outliers = DataCleaner.detect_outliers(cleaned, method="iqr")
print(outliers.to_string(index=False))
n_outliers = outliers.sum().sum()
print(f"共检测到 {n_outliers} 个异常值")

# 2c. 数据标准化
print("\n--- 数据标准化 (Min-Max) ---")
normalized = DataCleaner.normalize(cleaned, method="minmax")
print(normalized.round(4).to_string(index=False))

print("\n--- 数据标准化 (Z-Score) ---")
zscore = DataCleaner.normalize(cleaned, method="zscore")
print(zscore.round(4).to_string(index=False))

print("\n✅ 数据处理示例完成!")
