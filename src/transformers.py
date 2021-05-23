from typing import Tuple
import torch
import torch.nn as nn
from sparsemax import Sparsemax


class AttentiveTransformer(nn.Module):

    def __init__(self, input_size: int, rho: float = 1.0) -> None:
        super().__init__()
        self.h = nn.linear(input_size, input_size)
        self.bn = nn.BatchNorm2d()
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
        p = self.update_p(p)
        return mask, p


class FeatureBlock(nn.Module):

    def __init__(self, input_size: int, output_size: int = None,
                 sub_block_size: int = 2) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = input_size if not output_size else output_size
        self.sub_blocks = [self.feature_sub_block() for _ in range(sub_block_size)]

    def feature_sub_block(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.input_size, self.input_size),
            nn.BatchNorm2d,
            nn.GLU
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # TODO: double check whether it's bad to have skip conn on first sub block
        for sub_block in self.sub_blocks:
            identity = X
            out = sub_block(X)
            out += identity
            # Normalization which halves Var
            out *= torch.sqrt(0.5)
        return out


class FeatureTransformer(nn.Module):

    def __init__(self, input_size: int, nblocks: int = 2) -> None:
        super().__init__()
        self.input_size = input_size
        blocks = [FeatureBlock(input_size) for _ in range(nblocks)]
        self.blocks = nn.Sequential(*blocks)


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # TODO: check output dims expected for split block
        return self.blocks(X)


def sparsity_regularization(masks: torch.Tensor, eta: float = 0.00001) -> float:
    pass