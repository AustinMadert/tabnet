from typing import Tuple
import torch
import torch.nn as nn
from sparsemax import Sparsemax
from activations import GLU


class AttentiveTransformer(nn.Module):

    def __init__(self, hidden_dim: int, rho: float = 1.0) -> None:
        super().__init__()
        self.h = nn.Linear(hidden_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.sparsemax = Sparsemax(dim=-1)
        self.rho = rho


    def update_p(self, p: torch.Tensor, mask: torch.Tensor
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        update = self.rho - mask
        p *= update
        return p


    def forward(self, a: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        a = self.bn(self.h(a))
        a = p * a
        mask = self.sparsemax(a)
        p = self.update_p(p, mask)
        return mask, p


class FeatureBlock(nn.Module):

    def __init__(self, hidden_dim: int, output_dim: int = None,
                 num_sub_blocks: int = 2) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim if not output_dim else output_dim
        self.sub_blocks = [self.feature_sub_block() for _ in range(num_sub_blocks)]


    def feature_sub_block(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.BatchNorm1d(self.hidden_dim),
            GLU(input_dim=self.hidden_dim)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        for sub_block in self.sub_blocks:
            identity = X
            out = sub_block(X)
            out += identity
            # Normalization which halves Var
            out *= torch.sqrt(0.5)
        return out


def sparsity_regularization(masks: torch.Tensor, eta: float = 0.00001) -> float:
    pass