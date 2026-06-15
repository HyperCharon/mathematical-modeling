"""图论模块测试."""

import numpy as np
import pytest
from mathflow.graph import ShortestPath, MinSpanningTree, Hungarian, CPM


class TestShortestPath:
    def test_dijkstra(self):
        sp = ShortestPath()
        sp.add_edges([(0, 1, 4), (0, 2, 1), (2, 1, 2), (1, 3, 1)], directed=False)
        result = sp.dijkstra(0, 3)
        assert result.distance == 4  # 0→2→1→3

    def test_floyd(self):
        sp = ShortestPath()
        sp.add_edges([(0, 1, 4), (0, 2, 1), (2, 1, 2), (1, 3, 1)], directed=False)
        dist = sp.floyd()
        assert dist[0][3] == 4


class TestMinSpanningTree:
    def test_kruskal(self):
        mst = MinSpanningTree()
        mst.add_edges([(0, 1, 4), (0, 2, 1), (1, 2, 2), (1, 3, 5), (2, 3, 8)])
        result = mst.kruskal()
        assert result.total_weight == 8  # 1+2+5


class TestHungarian:
    def test_minimize(self):
        cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
        h = Hungarian(cost)
        result = h.solve()
        assert result.total_cost == 5  # (0,1)=1 + (1,0)=2 + (2,2)=2

    def test_maximize(self):
        profit = np.array([[10, 5, 8], [7, 9, 6], [8, 6, 10]])
        h = Hungarian(profit, maximize=True)
        result = h.solve()
        assert result.total_cost >= 25  # max profit


class TestCPM:
    def test_simple(self):
        cpm = CPM()
        cpm.add_activity("A", duration=3, predecessors=[])
        cpm.add_activity("B", duration=5, predecessors=[])
        cpm.add_activity("C", duration=4, predecessors=["A"])
        cpm.add_activity("D", duration=2, predecessors=["B"])
        cpm.add_activity("E", duration=3, predecessors=["C", "D"])
        result = cpm.solve()
        assert result.project_duration == 10  # B→D→E = 5+2+3 = 10
        assert "B" in result.critical_path
        assert "D" in result.critical_path
        assert "E" in result.critical_path


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
