import torch
import torch.nn as nn


class GLU(nn.Module):

    def __init__(self, n_d: int):
        super().__init__()
        self.fc1 = nn.Linear(n_d, n_d)
        self.fc2 = nn.Linear(n_d, n_d)
        self.sigmoid = nn.Sigmoid()


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.fc1(X) * self.sigmoid(self.fc2(X))
