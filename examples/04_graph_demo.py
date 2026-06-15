"""
示例 4: 图论 — 最短路径 + TSP

场景: 物流配送网络
"""

import sys
sys.path.insert(0, "..")

from mathflow.graph import ShortestPath, MinSpanningTree, TSPSolver
import numpy as np


def main():
    print("=" * 60)
    print("  示例: 图论算法 — 物流配送")
    print("=" * 60)

    # ======== 最短路径 ========
    print("\n【1】Dijkstra 最短路径")
    print("-" * 40)

    sp = ShortestPath()
    # 物流网络: (起点, 终点, 距离km)
    edges = [
        (0, 1, 4), (0, 2, 2), (1, 2, 1), (1, 3, 5),
        (2, 3, 8), (2, 4, 10), (3, 4, 2), (3, 5, 6),
        (4, 5, 3),
    ]
    sp.add_edges(edges, directed=False)

    # 求从仓库(0)到客户(5)的最短路径
    result = sp.dijkstra(0, 5)
    print(f"  最短距离: {result.distance:.1f} km")
    print(f"  最短路径: {result.path}")

    # 全源最短路
    all_dist = sp.floyd()
    print(f"\n  全源最短距离矩阵:")
    for i in range(len(all_dist)):
        print(f"    {i}: {[f'{d:.0f}' if d < 9999 else '∞' for d in all_dist[i]]}")

    # ======== 最小生成树 ========
    print("\n\n【2】最小生成树 (Kruskal)")
    print("-" * 40)

    mst = MinSpanningTree()
    mst.add_edges(edges)
    mst_result = mst.kruskal()
    print(mst.summary())

    # ======== TSP 旅行商 ========
    print("\n\n【3】TSP 旅行商问题")
    print("-" * 40)

    # 6个配送点的坐标
    coords = np.array([
        [0, 0],    # 仓库
        [2, 8],    # 客户1
        [5, 2],    # 客户2
        [7, 7],    # 客户3
        [9, 3],    # 客户4
        [3, 5],    # 客户5
    ])

    # 最近邻启发式
    tsp_nn = TSPSolver(coords)
    result_nn = tsp_nn.solve(method="nearest_neighbor")
    print(f"  最近邻: 路线={result_nn.route}, 距离={result_nn.total_distance:.2f}")

    # 2-opt 局部搜索
    tsp_2opt = TSPSolver(coords)
    result_2opt = tsp_2opt.solve(method="two_opt")
    print(f"  2-opt:  路线={result_2opt.route}, 距离={result_2opt.total_distance:.2f}")

    # 遗传算法
    tsp_ga = TSPSolver(coords)
    result_ga = tsp_ga.solve(method="genetic")
    print(f"  遗传算法: 路线={result_ga.route}, 距离={result_ga.total_distance:.2f}")

    # 绘图
    fig = tsp_2opt.plot(result_2opt)
    fig.savefig("04_tsp_route.png", dpi=150, bbox_inches="tight")
    print(f"\n  ✅ 已保存: 04_tsp_route.png")

    fig2 = sp.plot()
    fig2.savefig("04_network.png", dpi=150, bbox_inches="tight")
    print(f"  ✅ 已保存: 04_network.png")


if __name__ == "__main__":
    main()
