import torch
import torch.nn as nn
from decoder import TabNetDecoder
from encoder import TabNetEncoder
from transformers import FeatureBlock


class TabNet(nn.Module):

    def __init__(self, input_size: int, batch_size: int,  n_steps: int = 10, 
                output_size: int = None) -> None:
        super().__init__()
        self.input_size = input_size * 2
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.output_size = output_size
        self.shared_feat = self.build_shared_feature_transformer(input_size)
        self.encoder = TabNetEncoder(
            shared_feat=self.shared_feat,
            input_size=input_size,
            batch_size=batch_size,
            output_size=output_size,
            n_steps=n_steps
        )
        self.decoder = TabNetDecoder(
            shared_feat=self.shared_feat,
            input_size=input_size,
            batch_size=batch_size,
            output_size=output_size,
            n_steps=n_steps
        )


    def build_shared_feature_transformer(self, input_size) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.input_size, input_size),
            nn.BatchNorm2d(self.input_size),
            nn.GLU(),
            FeatureBlock(self.input_size, sub_block_size=1)
        )

    # TODO: need to add flags to control mode of operation, or build separate classes
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        encoded, attributes = self.encoder(X)
        X = self.decoder(encoded)
        return X, attributes