import torch
from torch import nn


class CNNClassifier(nn.Module):
    def __init__(self, input_channels: int, output_dim: int):
        super().__init__()
        self.conv = nn.Conv1d(input_channels, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)
