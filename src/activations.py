import torch
import torch.nn as nn


class GLU(nn.Module):

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Sigmoid()


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.fc1(X) * self.sigmoid(self.fc2(X))
        