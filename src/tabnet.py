import torch
import torch.nn as nn
from decoder import TabNetDecoder
from encoder import TabNetEncoder


class TabNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.encoder = TabNetEncoder()
        self.decoder = TabNetDecoder()


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.encoder(X)
        X = self.decoder(X)
        return X