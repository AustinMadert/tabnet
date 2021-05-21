import torch
import torch.nn as nn


class TabNetEncoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X


class TabNetDecoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X


class TabNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.encoder = TabNetEncoder()
        self.decoder = TabNetDecoder()


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.encoder(X)
        X = self.decoder(X)
        return X