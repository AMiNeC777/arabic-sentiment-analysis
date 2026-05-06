import torch
from torch import nn


class AraBERTClassifier(nn.Module):
    """Light classifier head placeholder for AraBERT outputs."""

    def __init__(self, hidden_size: int = 768, num_labels: int = 3):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, pooled_output):
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)
