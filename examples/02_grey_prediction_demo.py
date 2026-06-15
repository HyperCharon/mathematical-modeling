"""
示例 2: GM(1,1) 灰色预测

场景: 预测某城市未来3年的GDP
"""

import sys
sys.path.insert(0, "..")

from mathflow.predict import GreyPrediction


def main():
    print("=" * 60)
    print("  示例: GM(1,1) 灰色预测 — GDP 预测")
    print("=" * 60)

    # 某城市 GDP 数据 (亿元)
    gdp_data = [3200, 3580, 3890, 4250, 4680, 5120, 5650]

    print(f"\n  历史数据: {gdp_data}")
    print(f"  数据点数: {len(gdp_data)}")

    # 建模
    gp = GreyPrediction(gdp_data)
    gp.fit()
    print(gp.summary(steps=3))

    # 预测
    forecasts = gp.predict(steps=3)
    print(f"\n  未来3年 GDP 预测:")
    for i, f in enumerate(forecasts):
        print(f"    第{i+1}年: {f:.0f} 亿元")

    # 绘图
    fig = gp.plot(steps=3)
    fig.savefig("02_grey_prediction.png", dpi=150, bbox_inches="tight")
    print(f"\n  ✅ 已保存: 02_grey_prediction.png")


if __name__ == "__main__":
    main()
