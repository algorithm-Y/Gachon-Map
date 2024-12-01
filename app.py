from flask import Flask, render_template, request, jsonify
import os
from dijkstra import dijkstra
import networkx as nx
import matplotlib.pyplot as plt

app = Flask(__name__)

# Graph data
adj_matrix = [
    [0, 2, 0, 6, 0],
    [0, 0, 3, 8, 5],
    [0, 0, 0, 0, 7],
    [0, 0, 0, 0, 9],
    [0, 0, 0, 0, 0]
]

node_positions = {
    0: (466.67, 106.22),
    1: (266.67, 212.44),
    2: (426.67, 504.56),
    3: (693.33, 225.72),
    4: (666.67, 544.39)
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/shortest_path', methods=['POST'])
def shortest_path():
    data = request.json
    source = int(data['source'])
    target = int(data['target'])

    # Run Dijkstra algorithm
    dist, path = dijkstra(adj_matrix, source, target)

    # Save the visualization
    result_image_path = os.path.join('static', 'shortest_path.png')
    visualize_graph_with_background(adj_matrix, dist, path, result_image_path)

    # Return JSON response
    return jsonify({
        'distance': dist,
        'path': path,
        'image_path': 'static/shortest_path.png'
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
