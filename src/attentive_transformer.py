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