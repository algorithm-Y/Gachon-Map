from flask import Flask, render_template, request, jsonify
import os
from dijkstra import dijkstra  # Import the Dijkstra algorithm for shortest path computation
import networkx as nx          # Used for creating and visualizing graphs
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time as TIME            # Import time to generate unique filenames

app = Flask(__name__)          # Initialize the Flask application

# Graph data
adj_matrix = [
    [0.0, 61.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0],  # H
    [57.7, 0.0, 62.9, 0.0, 0.0, 0.0, 48.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 166.4,
     0.0],  # W
    [0.0, 61.9, 0.0, 112.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0],  # E
    [0.0, 0.0, 103.7, 0.0, 93.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0],  # C
    [0.0, 0.0, 0.0, 83.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 104.0,
     0.0],  # B
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0],
    # A
    [0.0, 47.7, 0.0, 0.0, 0.0, 0.0, 0.0, 69.7, 55.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0],  # J
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 68.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 88.2, 0.0, 0.0,
     0.0],  # K
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 55.5, 0.0, 0.0, 45.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0],  # L
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 36.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 54.3, 60.8, 103.2, 0.0,
     0.0],  # X
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 116.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0,
     0.0],  # P
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 73.2, 0.0, 69.6, 0.0,
     111.2],  # Y
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 106.1, 0.0, 0.0, 97.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0],  # Q
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 101.8, 0.0, 148.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.1],  # R
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 143.8, 0.0, 99.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0],  # S
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 95.0, 0.0, 159.1, 205.4, 0.0, 0.0, 0.0, 0.0,
     0.0],  # T
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 175.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0],  # V1
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 195.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0],  # V2
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 49.2, 0.0, 78.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0],  # α
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 81.0, 0.0, 60.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0],  # 7-10
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 97.3, 0.0, 72.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0],  # 9-5
    [0.0, 155.6, 0.0, 0.0, 98.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0],  # 1-5
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 97.2, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # 11-13
]

adj_matrix_mde = [
    [0.0, 61.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0],
    [57.7, 0.0, 62.9, 0.0, 0.0, 0.0, 48.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2,
     0.0],
    [0.0, 61.9, 0.0, 112.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0],
    [0.0, 0.0, 103.7, 0.0, 93.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0],
    [0.0, 0.0, 0.0, 83.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 104.0,
     0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0],
    [0.0, 20, 0.0, 0.0, 0.0, 0.0, 0.0, 69.7, 55.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 88.2, 0.0, 0.0,
     0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 55.5, 0.0, 0.0, 45.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 36.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 54.3, 60.8, 103.2, 0.0,
     0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50, 0.0, 70, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 159.1, 60, 0.0, 0.0, 0.0, 0.0,
     0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 175.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 60, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 49.2, 0.0, 78.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 140, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 140, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
]

node_positions = {
    0: (420, 70),  # H
    1: (475, 120),  # W
    2: (620, 120),  # E
    3: (750, 150),  # C
    4: (810, 190),  # B
    5: (680, 260),  # A
    6: (350, 170),  # J
    7: (260, 230),  # K
    8: (410, 220),  # L
    9: (490, 260),  # X
    10: (350, 380),  # P
    11: (650, 350),  # Y
    12: (410, 500),  # Q
    13: (670, 500),  # R
    14: (690, 600),  # S
    15: (550, 640),  # T
    16: (230, 510),  # V1
    17: (260, 620),  # V2
    18: (515, 350),  # α
    19: (310, 320),  # 7-10
    20: (630, 290),  # 9-5
    21: (700, 230),  # 1-5
    22: (720, 430)  # 11-13
}

# Route for the homepage
@app.route('/')
def index():
    # Render the initial webpage
    return render_template('index.html')


# Route to handle shortest path computation
@app.route('/shortest_path', methods=['POST'])
def shortest_path():
    data = request.json
    source = int(data['source']) # Source node
    target = int(data['target']) # Target node
    print(f"Received source: {source}, target: {target}")  # Log received input

    # Run Dijkstra algorithm(1st)
    time, path = dijkstra(adj_matrix, source, target)

    # Save the visualization with a unique name
    timestamp = int(TIME.time())  # Use current timestamp for uniqueness
    result_image_path = os.path.join('static', f'shortest_path_{timestamp}.png')

    # Run Dijkstra algorithm(2nd)
    time_mde, path_mde = dijkstra(adj_matrix_mde, source, target)

    # Visualize both graphs together
    visualize_graph_with_background(adj_matrix, time, path, adj_matrix_mde, time_mde, path_mde, result_image_path)

    # Return JSON response
    return jsonify({
        'time': time,
        'path': path,
        'time_mde': time_mde,
        'path_mde': path_mde,
        'image_path': result_image_path
    })


def visualize_graph_with_background(adj_matrix, time, path, adj_matrix_mde, time_mde, path_mde, output_path):
    G1 = nx.Graph()  # Create a graph for the first adjacency matrix
    G2 = nx.Graph()  # Create a graph for the second adjacency matrix
    exclude_nodes = {19, 20, 21, 22}  # Nodes to exclude from visualization

    # Build graph from adj_matrix
    for u in range(len(adj_matrix)):
        for v in range(u + 1, len(adj_matrix)):
            if adj_matrix[u][v] != 0:
                G1.add_edge(u, v, weight=adj_matrix[u][v])

    # Build graph from adj_matrix_mde
    for u in range(len(adj_matrix_mde)):
        for v in range(u + 1, len(adj_matrix_mde)):
            if adj_matrix_mde[u][v] != 0:
                G2.add_edge(u, v, weight=adj_matrix_mde[u][v])

    # Filter positions and labels for visualization
    filtered_pos = {node: pos for node, pos in node_positions.items()}
    visible_nodes = [node for node in G1.nodes if node not in exclude_nodes]
    filtered_labels = {node: str(node) for node in G1.nodes if node not in exclude_nodes}

    # Create positions and background image
    img = plt.imread('static/gachonMap.jpg')
    img_height, img_width, _ = img.shape

    plt.figure(figsize=(13, 9.3))
    plt.imshow(img, extent=[0, img_width, 0, img_height], aspect='auto')

    # Draw the first graph (adj_matrix) with its path
    nx.draw_networkx_nodes(G1, filtered_pos, nodelist=visible_nodes, node_size=700, node_color='skyblue')
    nx.draw_networkx_labels(G1, filtered_pos, labels=filtered_labels, font_size=12, font_weight='bold')

    weights1 = nx.get_edge_attributes(G1, 'weight')
    weights2 = nx.get_edge_attributes(G2, 'weight')

    # nx.draw_networkx_edge_labels(G1, filtered_pos, edge_labels=weights1, font_size=10)
    nx.draw_networkx_edges(G1, filtered_pos, edge_color='gray', width=2)
    # nx.draw_networkx_edge_labels(G2, filtered_pos, edge_labels=weights2, font_size=10)
    nx.draw_networkx_edges(G2, filtered_pos, edge_color='gray', width=2)



    # Highlight specific edges in the first graph
    if path:
        path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        nx.draw_networkx_edges(G1, filtered_pos, edgelist=path_edges, edge_color='red', width=4)


    # Highlight specific edges in the second graph
    if path_mde:
        path_mde_edges = [(path_mde[i], path_mde[i + 1]) for i in range(len(path_mde) - 1)]

        # Highlight edges in path_mde that are also in specific_edges
        specific_edges = [(1, 21), (21, 5), (5, 20), (20, 11), (11, 22), (22, 13), (13, 14), (14, 15), (15, 17),
                          (17, 15), (15, 14), (14, 13), (13, 12), (12, 10), (10, 19), (19, 7), (7, 6),
                          (6, 1), ]  # Example: manually specify edges
        specific_edges_in_path = [edge for edge in specific_edges if edge in path_mde_edges]

        # Draw specific edges in blue
        nx.draw_networkx_edges(G2, filtered_pos, edgelist=specific_edges_in_path, edge_color='blue', width=2)

    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    app.run(debug=True) # Run the Flask app in debug mode