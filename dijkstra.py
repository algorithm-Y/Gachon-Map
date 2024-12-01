import heapq

def dijkstra(adj_matrix, source, target):
    V = len(adj_matrix)
    dist = [float('inf')] * V
    dist[source] = 0
    prev = [None] * V
    visited = [False] * V
    min_heap = [(0, source)]

    # Main loop: Repeat until minimum hip is full
    while min_heap:
        current_dist, u = heapq.heappop(min_heap)    # Least distance node pulled from heap
        if visited[u]:                               # Skip nodes already visited
            continue
        visited[u] = True                            # Visit and process current nodes

        # Explore all neighboring nodes v of the current node u
        for v in range(V):
            if adj_matrix[u][v] != 0:
                alt = current_dist + adj_matrix[u][v]
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u
                    heapq.heappush(min_heap, (dist[v], v))
    # Path tracking: creation of a path from the target node to the start node
    path = []
    while target is not None:
        path.insert(0, target)
        target = prev[target]

    # Return Results
    if dist[path[-1]] != float('inf'):
        return dist[path[-1]], path
    return None, []
