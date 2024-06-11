import networkx as nx
import numpy as np

class DummyPongEnv:
    def __init__(self):
        self.width = 800
        self.height = 600
        self.proximity_threshold = 50  # Example threshold for proximity
        
        # Dummy positions for the objects
        self.ball = {'x': 100, 'y': 100}
        self.left_paddle = {'x': 50, 'y': 200}
        self.right_paddle = {'x': 750, 'y': 200}
        self.top_wall = {'x': 0, 'y': 0}
        self.bottom_wall = {'x': 0, 'y': self.height}

    def get_graph_data(self):
        # Initialize a NetworkX graph
        graph = nx.Graph()

        # Define object features and add nodes
        ball_features = [self.ball['x'], self.ball['y']]
        graph.add_node("ball", type="object", features=ball_features)
        
        left_paddle_features = [self.left_paddle['x'], self.left_paddle['y']]
        graph.add_node("left_paddle", type="object", features=left_paddle_features)
        
        right_paddle_features = [self.right_paddle['x'], self.right_paddle['y']]
        graph.add_node("right_paddle", type="object", features=right_paddle_features)
        
        top_wall_features = [self.top_wall['x'], self.top_wall['y']]
        graph.add_node("top_wall", type="object", features=top_wall_features)
        
        bottom_wall_features = [self.bottom_wall['x'], self.bottom_wall['y']]
        graph.add_node("bottom_wall", type="object", features=bottom_wall_features)

        # Combine object positions
        object_positions = {
            "ball": ball_features,
            "left_paddle": left_paddle_features,
            "right_paddle": right_paddle_features,
            "top_wall": top_wall_features,
            "bottom_wall": bottom_wall_features
        }

        # Proximity threshold for creating atoms
        proximity_threshold = self.proximity_threshold

        # Create atom nodes and edges based on proximity
        atom_index = 0  # Index to differentiate atom nodes
        for i, (obj1, pos1) in enumerate(object_positions.items()):
            for j, (obj2, pos2) in enumerate(object_positions.items()):
                if i < j:
                    dist = np.linalg.norm(np.array(pos1) - np.array(pos2))
                    if dist < proximity_threshold:
                        # Create an atom node for proximity
                        atom_node = f"proximity_{atom_index}"
                        graph.add_node(atom_node, type="atom", predicate="close_to")

                        # Add edges from the atom node to the object nodes
                        graph.add_edge(obj1, atom_node, position=0)  # Position 0 in the predicate
                        graph.add_edge(obj2, atom_node, position=1)  # Position 1 in the predicate

                        atom_index += 1  # Increment atom index for the next atom

        return graph

# Create the dummy environment and get the graph data
env = DummyPongEnv()
graph = env.get_graph_data()

# Display the graph for inspection
for node in graph.nodes(data=True):
    print(node)

for edge in graph.edges(data=True):
    print(edge)