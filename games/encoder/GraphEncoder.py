import torch
import networkx as nx
from itertools import combinations
from collections import defaultdict
from typing import NamedTuple
from torch_geometric.data import HeteroData, Batch

class PredicateEdgeType(NamedTuple):
    src_type: str
    pos: str
    dst_type: str

import torch
import networkx as nx
from torch_geometric.data import HeteroData
from collections import defaultdict
from itertools import combinations

class HeteroGNNEncoder:
    def __init__(self, obj_type_id: str = "obj", atom_type_id: str = "atom"):
        self.obj_type_id = obj_type_id
        self.atom_type_id = atom_type_id

    def encode(self, batch_node_features: torch.Tensor, proximity_threshold: float = 1000) -> HeteroData:
        batch_data = []
        batch_size = batch_node_features.size(0)

        for b in range(batch_size):
            node_features = batch_node_features[b]
            num_nodes = node_features.size(0)
            graph = nx.Graph()

            # Adding object nodes
            for i in range(num_nodes):
                graph.add_node(i, type=self.obj_type_id, features=node_features[i].tolist())

            # Adding atom nodes
            atom_index = num_nodes
            object_feature_length = node_features.size(1)
            for i, j in combinations(range(num_nodes), 2):
                dist = torch.norm(node_features[i, :2] - node_features[j, :2]).item()
                if dist < proximity_threshold:
                    # Create atom node with a 2D zero vector of the shape (2, object_feature_length)
                    atom_features = torch.zeros((2, object_feature_length)).tolist()
                    graph.add_node(atom_index, type=self.atom_type_id, features=atom_features)
                    graph.add_edge(i, atom_index, position=0)
                    graph.add_edge(j, atom_index, position=1)
                    atom_index += 1

            batch_data.append(graph)

        return Batch.from_data_list(self.to_pyg_data(batch_data))

    def to_pyg_data(self, batch_graphs):
        data_list = []

        for graph in batch_graphs:
            data = HeteroData()
            node_index_mapping = {self.obj_type_id: {}, self.atom_type_id: {}}
            obj_features = []
            atom_features = []
            edge_dict = defaultdict(list)

            current_obj_features = []
            current_atom_features = []

            for node, attrs in graph.nodes(data=True):
                node_type = attrs['type']
                features = torch.tensor(attrs['features'])
                if node_type == self.obj_type_id:
                    node_index_mapping[node_type][node] = len(current_obj_features)
                    current_obj_features.append(features)
                elif node_type == self.atom_type_id:
                    node_index_mapping[node_type][node] = len(current_atom_features)
                    current_atom_features.append(features)

            if current_obj_features:
                obj_features.append(torch.stack(current_obj_features))
            if current_atom_features:
                flattened_atom_features = [f.view(-1) for f in current_atom_features]
                atom_features.append(torch.stack(flattened_atom_features))

            if obj_features:
                data[self.obj_type_id].x = torch.cat(obj_features)
            if atom_features:
                data[self.atom_type_id].x = torch.cat(atom_features)

            for src, dst, attr in graph.edges(data=True):
                src_type = graph.nodes[src]['type']
                dst_type = graph.nodes[dst]['type']
                pos = str(attr['position'])
                edge_type = (src_type, pos, dst_type)

                src_idx = node_index_mapping[src_type][src]
                dst_idx = node_index_mapping[dst_type][dst]
                edge_dict[edge_type].append((src_idx, dst_idx))
                # Add reverse edges for bidirectionality
                reverse_edge_type = (dst_type, pos, src_type)
                edge_dict[reverse_edge_type].append((dst_idx, src_idx))

            for edge_type, edges in edge_dict.items():
                edge_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()
                data[edge_type].edge_index = edge_tensor

            data_list.append(data)

        return data_list  


# Example usage

