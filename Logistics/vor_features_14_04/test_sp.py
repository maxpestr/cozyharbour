import networkx as nx
import numpy as np

def build_shortest_paths(weighted_graph):
        # Создаем направленный граф из взвешенного графа
        G = nx.DiGraph()
        for source, targets in weighted_graph.items():
            for target, weight in targets.items():
                G.add_edge(source, target, weight=weight)
        
        # Находим кратчайшие пути между всеми парами вершин с помощью алгоритма Флойда-Уоршалла
        matrix = nx.floyd_warshall_numpy(G)
        
        return matrix
# Создадим случайный взвешенный граф
weighted_graph = {
    'A': {'B': 1},
    'A': {'C': 0.51},
    'B': {'C': 1},
    'C': {'D': 1},
    'D': {'A': 2}
}


# Получим матрицу кратчайших путей
matrix = build_shortest_paths(weighted_graph)

# Выведем матрицу кратчайших путей
print(matrix)