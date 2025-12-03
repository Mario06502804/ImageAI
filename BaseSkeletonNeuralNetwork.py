import torch
import torch.nn as nn
import torch.nn.functional as f

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # Define Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 10)

        