from flask import Flask, render_template, request, jsonify
import os
from dijkstra import dijkstra
import networkx as nx
import matplotlib.pyplot as plt
import time

app = Flask(__name__)

# Graph data
adj_matrix = [
    [0.0 , 61.1 , 0.0  , 0.0  , 0.0  , 0.0  , 0.0 , 0.0 , 0.0 , 0.0 , 0.0  , 0.0 , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0],  # H
    [57.7, 0.0  , 62.9 , 0.0  , 0.0  , 166.4, 48.5, 0.0 , 0.0 , 0.0 , 0.0  , 0.0 , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0],  # W
    [0.0 , 61.9 , 0.0  , 112.9, 0.0  , 0.0  , 0.0 , 0.0 , 0.0 , 0.0 , 0.0  , 0.0 , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0],  # E
    [0.0 , 0.0  , 103.7, 0.0  , 93.8 , 0.0  , 0.0 , 0.0 , 0.0 , 0.0 , 0.0  , 0.0 , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0],  # C
    [0.0 , 0.0  , 0.0  , 83.5 , 0.0  , 104.0, 0.0 , 0.0 , 0.0 , 0.0 , 0.0  , 0.0 , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0],  # B
    [0.0 , 155.6, 0.0  , 0.0  , 98.2 , 0.0  , 0.0 , 0.0 , 0.0 , 97.3, 0.0  , 72.1, 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0],  # A
    [0.0 , 47.7 , 0.0  , 0.0  , 0.0  , 0.0  , 0.0 , 69.7, 55.5, 0.0 , 0.0  , 0.0 , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0],  # J
    [0.0 , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 68.6, 0.0 , 0.0 , 0.0 , 88.2 , 0.0 , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0],  # K
    [0.0 , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 55.5, 0.0 , 0.0 , 45.0, 0.0  , 0.0 , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0],  # L
    [0.0 , 0.0  , 0.0  , 0.0  , 0.0  , 103.2, 0.0 , 0.0 , 36.4, 0.0 , 60.8 , 0.0 , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 54.3],  # X
    [0.0 , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0 , 81.0, 0.0 , 60.8, 0.0  , 0.0 , 116.7, 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0],  # P
    [0.0 , 0.0  , 0.0  , 0.0  , 0.0  , 69.6 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0  , 0.0 , 0.0  , 111.2, 0.0  , 0.0  , 0.0  , 0.0  , 73.2], # Y
    [0.0 , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0 , 0.0 , 0.0 , 0.0 , 106.1, 0.0 , 0.0  , 97.9 , 0.0  , 0.0  , 0.0  , 0.0  , 0.0],  # Q
    [0.0 , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0 , 0.0 , 0.0 , 0.0 , 0.0  , 97.2, 101.8, 0.0  , 148.5, 0.0  , 0.0  , 0.0  , 0.0],  # R
    [0.0 , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0 , 0.0 , 0.0 , 0.0 , 0.0  , 0.0 , 0.0  , 143.8, 0.0  , 99.7 , 0.0  , 0.0  , 0.0],  # S
    [0.0 , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0 , 0.0 , 0.0 , 0.0 , 0.0  , 0.0 , 0.0  , 0.0  , 95.0 , 0.0  , 159.1, 205.4, 0.0],  # T
    [0.0 , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0 , 0.0 , 0.0 , 0.0 , 0.0  , 0.0 , 0.0  , 0.0  , 0.0  , 175.0, 0.0  , 0.0  , 0.0],  # V1
    [0.0 , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0 , 0.0 , 0.0 , 0.0 , 0.0  , 0.0 , 0.0  , 0.0  , 0.0  , 195.0, 0.0  , 0.0  , 0.0],  # V2
    [0.0 , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0 , 0.0 , 0.0 , 49.2, 0.0  , 78.1, 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0  , 0.0]   # α
]

node_positions = {
    0: (420, 70),   # H --
    1: (475, 120),   # W --
    2: (620, 120),   # E --
    3: (750, 150),   # C --
    4: (840, 240),   # B --
    5: (720, 350),   # A --
    6: (350, 170),   # J --
    7: (230, 240),   # K --
    8: (410, 220),   # L --
    9: (475, 240),   # X -
    10: (300, 420),  # P --
    11: (650, 350), # Y --
    12: (390, 530),  # Q -
    13: (650, 520),  # R -
    14: (700, 610),  # S -
    15: (550, 640),  # T
    16: (230, 510),  # V1
    17: (220, 610),   # V2
    18: (515, 350),  # α
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/shortest_path', methods=['POST'])
def shortest_path():
    data = request.json
    source = int(data['source'])
    target = int(data['target'])
    print(f"Received source: {source}, target: {target}")  # 로그 출력

    # Run Dijkstra algorithm
    dist, path = dijkstra(adj_matrix, source, target)

    # Save the visualization with a unique name
    timestamp = int(time.time())  # Use current timestamp for uniqueness
    result_image_path = os.path.join('static', f'shortest_path_{timestamp}.png')
    visualize_graph_with_background(adj_matrix, dist, path, result_image_path)

    # Return JSON response
    return jsonify({
        'distance': dist,
        'path': path,
        'image_path': result_image_path
    })

def visualize_graph_with_background(adj_matrix, dist, path, output_path):
    G = nx.Graph()
    for u in range(len(adj_matrix)):
        for v in range(u + 1, len(adj_matrix)):
            if adj_matrix[u][v] != 0:
                G.add_edge(u, v, weight=adj_matrix[u][v])

    img = plt.imread('static/gachonMap.jpg')
    img_height, img_width, _ = img.shape
    pos = node_positions

    plt.figure(figsize=(12, 8))
    plt.imshow(img, extent=[0, img_width, 0, img_height], aspect='auto')
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue')
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    weights = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weights, font_size=10)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='gray', width=2)
    if path:
        path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=3)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()



if __name__ == '__main__':
    app.run(debug=True)
