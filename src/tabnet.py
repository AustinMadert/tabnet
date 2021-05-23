import torch
import torch.nn as nn
from decoder import TabNetDecoder
from encoder import TabNetEncoder


class TabNet(nn.Module):

    def __init__(self, input_size: int, batch_size: int,  n_steps: int = 10, 
                output_size: int = None) -> None:
        super().__init__()
        self.encoder = TabNetEncoder()
        self.decoder = TabNetDecoder()


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        encoded, attributes, shared_feat = self.encoder(X)
        X = self.decoder(X)
        return X