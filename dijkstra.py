import heapq

def dijkstra(adj_matrix, source, target):
    V = len(adj_matrix)
    dist = [float('inf')] * V
    dist[source] = 0
    prev = [None] * V
    visited = [False] * V
    min_heap = [(0, source)]

    while min_heap:
        current_dist, u = heapq.heappop(min_heap)
        if visited[u]:
            continue
        visited[u] = True
        for v in range(V):
            if adj_matrix[u][v] != 0:
                alt = current_dist + adj_matrix[u][v]
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u
                    heapq.heappush(min_heap, (dist[v], v))

    path = []
    while target is not None:
        path.insert(0, target)
        target = prev[target]

    if dist[path[-1]] != float('inf'):
        return dist[path[-1]], path
    return None, []
