import networkx as nx

def get_covered_nodes(object_rect, node_width, node_height, grid_width, grid_height):
    start_x = max(0, object_rect.left // node_width)
    end_x = min(grid_width - 1, object_rect.right // node_width)
    start_y = max(0, object_rect.top // node_height)
    end_y = min(grid_height - 1, object_rect.bottom // node_height)
    return [(x, y) for x in range(start_x, end_x + 1) for y in range(start_y, end_y + 1)]

def normalize_speed(speed, max_speed):
    return speed / max_speed

def calculate_manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def create_edges_with_distances(grid_width, grid_height):
    edges = []
    for x1 in range(grid_width):
        for y1 in range(grid_height):
            node1_index = y1 * grid_width + x1
            for x2 in range(grid_width):
                for y2 in range(grid_height):
                    node2_index = y2 * grid_width + x2
                    if node1_index != node2_index:  # Skip self-connections
                        distance = calculate_manhattan_distance(x1, y1, x2, y2)
                        edges.append((node1_index, node2_index, distance))
    return edges

def update_node_features(nodes, ball, left_paddle, right_paddle, ball_speed_x, ball_speed_y,left_paddle_move, right_paddle_move, paddle_speed,node_width, node_height, grid_width, grid_height):
    # Reset all node features
    for x in range(grid_width):
        for y in range(grid_height):
            node = nodes[x][y]
            # Reset features for simplicity in explanation; consider optimizing
            node['features'] = [0] * 18  # Adjusted for a fixed feature vector length
            
            # Wall presence
            node['features'][9] = 1 if y == 0 else 0  # Top wall
            node['features'][10] = 1 if y == grid_height - 1 else 0  # Bottom wall
            node['features'][11] = 1 if x == 0 else 0  # Left wall
            node['features'][12] = 1 if x == grid_width - 1 else 0  # Right wall
    # Update for ball
    for x, y in get_covered_nodes(ball, node_width, node_height, grid_width, grid_height):
        nodes[x][y]['features'][0] = 1  # Ball presence
        # Update velocity and other features as needed
        nodes[x][y]['features'][1] = normalize_speed(ball_speed_x,4)  # Normalized ball velocity X
        nodes[x][y]['features'][2] = normalize_speed(ball_speed_y,4)  # Normalized ball velocity Y
    # Update for left paddle spanning multiple nodes
    for x, y in get_covered_nodes(left_paddle, node_width, node_height, grid_width, grid_height):
        nodes[x][y]['features'][3] = 1  # Left paddle presence
        if left_paddle_move == -1:
            nodes[x][y]['features'][4] = -1
            # set paddle speed
            nodes[x][y]['features'][5] = paddle_speed
    # Update for right paddle spanning multiple nodes
    for x, y in get_covered_nodes(right_paddle, node_width, node_height,grid_width, grid_height):
        nodes[x][y]['features'][6] = 1  # Right paddle presence
        if right_paddle_move == -1:
            nodes[x][y]['features'][7] = -1
            # set paddle speed
            nodes[x][y]['features'][8] = paddle_speed
        elif right_paddle_move == 1:
            nodes[x][y]['features'][7] = 1
            nodes[x][y]['features'][8] = paddle_speed

    # Further feature updates can be added here, such as direction, nearby objects, etc.
    
        # Check adjacency for ball, paddles, or walls
        # Resetting or initializing adjacency indicators
    for x in range(grid_width):
        for y in range(grid_height):
            node = nodes[x][y]
            # Resetting or initializing adjacency indicators
            adjacent_features = [0] * 5
        for adj_x in range(max(0, x-1), min(x+2, grid_width)):
            for adj_y in range(max(0, y-1), min(y+2, grid_height)):
                    if adj_x == x and adj_y == y:
                        continue  # Skip the node itself

                    adj_node = nodes[adj_x][adj_y]
                    if adj_node['features'][0]:  # Ball presence
                        adjacent_features[0] = 1
                    if adj_node['features'][3] or adj_node['features'][6]:  # Paddle presence
                        adjacent_features[1] = 1
                    if adj_y == 0:  # Top wall adjacency
                        adjacent_features[2] = 1
                    if adj_y == grid_height - 1:  # Bottom wall adjacency
                        adjacent_features[3] = 1
                    if adj_x == 0 or adj_x == grid_width - 1:  # Side (left/right) wall adjacency
                        adjacent_features[4] = 1

                # Update the node features with adjacency information
            node['features'][-5:] = adjacent_features
    return nodes

def create_initial_graph(grid_width, grid_height):
    G = nx.Graph()
    # Add nodes
    for x in range(grid_width):
        for y in range(grid_height):
            node_id = y * grid_width + x
            G.add_node(node_id)  # Initially, features might not be added

    # Add edges based on your criteria (fully connected in this case)
    for node1 in G.nodes():
        for node2 in G.nodes():
            if node1 != node2:
                G.add_edge(node1, node2)
                # Optionally, add weights or other attributes to edges here

    return G
# In the game loop, after moving the ball and paddles
def create_initial_graph(grid_width, grid_height):
    G = nx.Graph()
    # Add nodes
    for x in range(grid_width):
        for y in range(grid_height):
            node_id = y * grid_width + x
            G.add_node(node_id)  # Initially, features might not be added

    # Add edges based on your criteria (fully connected in this case)
    for node1 in G.nodes():
        for node2 in G.nodes():
            if node1 != node2:
                G.add_edge(node1, node2)
                # Optionally, add weights or other attributes to edges here

    return G

def update_node_features_graph(G, nodes, grid_width, grid_height):
    # Assuming 'nodes' is a 2D list of dictionaries with updated features
    for x in range(grid_width):
        for y in range(grid_height):
            node_id = y * grid_width + x
            node_features = nodes[x][y]['features']
            # Update the node attributes in the graph
            nx.set_node_attributes(G, {node_id: {'features': node_features}})
