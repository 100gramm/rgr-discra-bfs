import matplotlib.pyplot as plt
import networkx as nx
from collections import deque
import numpy as np
import random

# ----------------------------------------------------------
# ЧТЕНИЕ МАТРИЦЫ ИЗ ФАЙЛА
# ----------------------------------------------------------
def read_matrix_from_file(filename):
    matrix = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            row = list(map(int, line.split()))
            matrix.append(row)
    return matrix


# ----------------------------------------------------------
# СЛУЧАЙНАЯ ГЕНЕРАЦИЯ ГРАФА
# ----------------------------------------------------------
def generate_random_graph(n, edges=None):
    print(f"Генерируем случайный граф из {n} вершин...")

    max_edges = n * (n - 1) // 2
    if edges is None:
        edges = random.randint(0, max_edges)

    print(f"Количество рёбер: {edges}")

    adj = np.zeros((n, n), dtype=int)

    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(all_edges)

    for i in range(edges):
        u, v = all_edges[i]
        adj[u][v] = adj[v][u] = 1

    print("Сгенерированная матрица смежности:")
    for row in adj:
        print(*row)

    return adj


# ----------------------------------------------------------
# ПРЕОБРАЗОВАНИЕ К СМЕЖНОСТИ
# ----------------------------------------------------------
def prepare_adjacency(matrix):
    matrix = np.array(matrix)
    n_rows, n_cols = matrix.shape

    if n_rows == n_cols:
        print("Входная матрица — матрица смежности.")
        return matrix
    else:
        print("Входная матрица — матрица инцидентности, преобразуем в смежность...")
        adj_matrix = np.zeros((n_rows, n_rows), dtype=int)

        for e in range(n_cols):
            vertices_in_edge = np.where(matrix[:, e] == 1)[0]
            if len(vertices_in_edge) == 2:
                u, v = vertices_in_edge
                adj_matrix[u][v] = adj_matrix[v][u] = 1

        print("Преобразованная матрица смежности:")
        for row in adj_matrix:
            print(*row)

        return adj_matrix


# ----------------------------------------------------------
# BFS
# ----------------------------------------------------------
def bfs_shortest_paths(adj_matrix, start=0):
    n = len(adj_matrix)
    visited = [False] * n
    distance = [None] * n
    parent = [None] * n
    order = []

    queue = deque([start])
    visited[start] = True
    distance[start] = 0

    while queue:
        u = queue.popleft()
        order.append(u)

        for v, connected in enumerate(adj_matrix[u]):
            if connected and not visited[v]:
                visited[v] = True
                parent[v] = u
                distance[v] = distance[u] + 1
                queue.append(v)

    return parent, distance, order


# ----------------------------------------------------------
# ВЫВОД ТАБЛИЦЫ
# ----------------------------------------------------------
def print_table(vertices, parent, distance, order):
    print("\nВершина | Родитель | Расстояние | Порядок обхода")
    print("-----------------------------------------------")
    for idx in order:
        parent_char = vertices[parent[idx]] if parent[idx] is not None else "-"
        print(f"{vertices[idx]:^8} | {parent_char:^8} | {distance[idx]:^10} | {order.index(idx)+1:^14}")


# ----------------------------------------------------------
# АНИМАЦИЯ BFS
# ----------------------------------------------------------
def animate_bfs(adj_matrix, vertices, parent, order):
    G = nx.Graph()
    n = len(adj_matrix)

    G.add_nodes_from(vertices)

    for i in range(n):
        for j in range(i, n):
            if adj_matrix[i][j]:
                G.add_edge(vertices[i], vertices[j])

    pos = nx.spring_layout(G, seed=42)

    node_colors = ['lightgrey'] * n
    edge_colors = ['lightgrey'] * G.number_of_edges()

    edge_list = list(G.edges())
    edge_index = {edge: idx for idx, edge in enumerate(edge_list)}
    edge_index.update({(v, u): idx for (u, v), idx in edge_index.items()})

    plt.ion()
    fig, ax = plt.subplots()

    for u in order:
        node_colors[u] = 'skyblue'

        if parent[u] is not None:
            u_name = vertices[u]
            p_name = vertices[parent[u]]
            idx = edge_index[(p_name, u_name)]
            edge_colors[idx] = 'red'

        ax.clear()
        nx.draw(G, pos, with_labels=True, node_color=node_colors,
                edge_color=edge_colors, node_size=800, width=3, ax=ax)
        plt.pause(0.7)

        node_colors[u] = 'lightgreen'
        if parent[u] is not None:
            edge_colors[idx] = 'lightgreen'

    plt.ioff()
    plt.show()


# ----------------------------------------------------------
# ОСНОВНАЯ ПРОГРАММА
# ----------------------------------------------------------
if __name__ == "__main__":
    print("Выберите способ ввода графа:")
    print("1 — Загрузить граф из файла")
    print("2 — Сгенерировать случайный граф")

    choice = input("Ваш выбор: ").strip()

    if choice == "1":
        filename = input("Введите имя файла: ")
        input_matrix = read_matrix_from_file(filename)

    else:
        n_input = input("Количество вершин (Enter — случайно 1–20): ").strip()
        if n_input == "":
            n = random.randint(1, 20)
            print(f"Случайное количество вершин: {n}")
        else:
            n = int(n_input)

        e_input = input("Количество рёбер (Enter — случайно): ").strip()
        e = int(e_input) if e_input else None

        input_matrix = generate_random_graph(n, e)

    adj_matrix = prepare_adjacency(input_matrix)

    n = len(adj_matrix)
    vertices = [chr(ord('A') + i) for i in range(n)]

    start_vertex = 0
    parent, distance, order = bfs_shortest_paths(adj_matrix, start=start_vertex)

    print_table(vertices, parent, distance, order)
    animate_bfs(adj_matrix, vertices, parent, order)
