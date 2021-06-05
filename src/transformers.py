from typing import Tuple
import torch
import torch.nn as nn
from sparsemax import Sparsemax
from activations import GLU
from normailzation import GhostBatchNormalization


class AttentiveTransformer(nn.Module):

    def __init__(self, batch_dim: int, n_d: int, rho: float = 1.0) -> None:
        super().__init__()
        self.p = torch.ones(batch_dim, n_d)
        self.h = nn.Linear(n_d, n_d)
        self.bn = GhostBatchNormalization(n_d, 8)
        self.sparsemax = Sparsemax(dim=-1)
        self.rho = rho


    def update_p(self, mask: torch.Tensor) -> None:
        update = self.rho - mask
        self.p *= update


    def forward(self, a: torch.Tensor) -> torch.Tensor:
        a = self.bn(self.h(a))
        a = self.p * a
        mask = self.sparsemax(a)
        self.update_p(mask)
        return mask


class FeatureBlock(nn.Module):

    def __init__(self, hidden_dim: int, num_sub_blocks: int = 2) -> None:
        super().__init__()
        self.sub_blocks = [self.feature_sub_block(hidden_dim) for _ in range(num_sub_blocks)]
        self.norm = torch.sqrt(torch.Tensor([0.5]))


    def feature_sub_block(self, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            GLU(n_d=hidden_dim)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        for sub_block in self.sub_blocks:
            identity = X
            out = sub_block(X)
            out += identity
            # Normalization which halves Var
            out *= self.norm
        return out


def sparsity_regularization(masks: torch.Tensor, eta: float = 0.00001) -> float:
    pass