import torch
import torch.nn as nn
from sparsemax import Sparsemax


class AttentiveTransformer(nn.Module):

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.h = nn.linear(input_size, input_size)
        self.bn = nn.BatchNorm2d()
        self.sparse_max = Sparsemax(dim=-1)
        self.prior_scales = torch.ones(input_size, input_size)


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.h(X)
        X = self.bn(X)
        
        out = self.sparse_max(X)
        return out