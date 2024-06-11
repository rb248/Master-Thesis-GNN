import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import numpy as np

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
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

        # Dummy input to calculate flat features size
        with torch.no_grad():
            dummy_input = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(dummy_input).shape[1]
            print(f"Calculated flat features size: {n_flatten}")

        self.adjust_to_features_dim = nn.Linear(n_flatten, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        cnn_output = self.cnn(observations)
        return self.adjust_to_features_dim(cnn_output)

# Assuming the observation space is a 3-channel image of size 84x84
observation_space = spaces.Box(low=0, high=255, shape=(3, 84, 84), dtype=np.uint8)

# Create a dummy observation
dummy_obs = observation_space.sample()[None]  # Add batch dimension
dummy_obs = torch.tensor(dummy_obs).float()  # Convert to float tensor

# Initialize the CNN
features_dim = 64  # This should match your downstream task requirements
cnn = CustomCNN(observation_space, features_dim)

# Run a forward pass
output = cnn(dummy_obs)
print("Output shape from CNN: ", output.shape)

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import numpy as np

def main():
    observation_space = spaces.Box(low=0, high=255, shape=(3, 84, 84), dtype=np.uint8)
    dummy_obs = observation_space.sample()[None]  # Add batch dimension
    dummy_obs = torch.tensor(dummy_obs).float()  # Convert to float tensor

    features_dim = 64
    cnn = CustomCNN(observation_space, features_dim)

    output = cnn(dummy_obs)
    print("Output shape from CNN: ", output.shape)

if __name__ == "__main__":
    main()
