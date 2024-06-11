import torch as th
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from games.model.hetero_gnn import HeteroGNN
from typing import Dict
from games.encoder.GraphEncoder import HeteroGNNEncoder
from gymnasium import spaces
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from games.encoder.GraphEncoder import HeteroGNNEncoder
from games.model.hetero_gnn import HeteroGNN
import torch_geometric as pyg
from games.model.cnn_model import CNNgame

class CustomHeteroGNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64, hidden_size=64, num_layer=2, obj_type_id='obj', arity_dict={'atom': 2}):
        super().__init__(observation_space, features_dim=hidden_size)
        self.encoder = HeteroGNNEncoder()
        self.model = HeteroGNN(hidden_size, num_layer, obj_type_id, arity_dict)

    def forward(self, observations):
        # Encode observations to a graph using the encoder
        pyg_data = self.encoder.encode(observations) 
        obj_emb = self.model(pyg_data.x_dict, pyg_data.edge_index_dict, pyg_data.batch_dict)
        # Flatten or pool the embeddings if necessary to match the expected features_dim
        return obj_emb



import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import numpy as np

# Checking the output dimension right before feeding into the Linear layer
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        n_input_channels = 4  # should be 4
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy_input = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(dummy_input).shape[1]
            print(f"Calculated flat features size: {n_flatten}")  # Debugging line

        self.adjust_to_features_dim = nn.Linear(2560, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        #check first dimension of the input tensor
        # if observations.shape[0] >=1:
        #     # take the last observation
        #     observations = observations[-1] 
        cnn_output = self.cnn(observations)
        final_output = self.adjust_to_features_dim(cnn_output)
        return final_output

