import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class CNNgame(nn.Module):
    def __init__(self):
        super(CNNgame, self).__init__()
        # Define the convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, 5, 1, 2)
        
        # Calculate the output size after convolution and pooling layers
        # Assume input size is (4, 600, 800)
        self._to_linear = self.calculate_conv_output((4, 600, 800))

        # Linear layers for feature flattening
        self.fc1 = nn.Linear(self._to_linear, 256)  # Adjust size based on actual output from conv layers

        # Linear layers for policy and value output
        self.fc_policy = nn.Linear(256, 3)  # Assuming 3 actions
        self.fc_value = nn.Linear(256, 1)

    def calculate_conv_output(self, shape):
        # Create a dummy tensor with the given shape
        dummy_input = torch.zeros(1, *shape)
        dummy_output = self.pool(F.relu(self.conv1(dummy_input)))
        dummy_output = self.pool(F.relu(self.conv2(dummy_output)))
        return int(np.prod(dummy_output.size()))

    def forward(self, current):
        # Process current state to extract features
        if isinstance(current, tuple):
            current = torch.cat(current, dim=1)  # Concatenate tensors along the channel dimension
        current = self.pool(F.relu(self.conv1(current)))
        current = self.pool(F.relu(self.conv2(current)))
        current = current.view(-1, self._to_linear)
        current_features = F.relu(self.fc1(current))

        # Calculate logits for the current state
        logits = self.fc_policy(current_features)

        # Apply softmax to compute probabilities from logits
        

        return logits
