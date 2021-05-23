import torch
import torch.nn as nn


class TabNetDecoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()


    def build_feature_transformer(self) -> nn.Module:
        return nn.Sequential(
            self.shared_feat,
            FeatureBlock(self.input_size)
        )


    def forward(self, X: torch.Tensor, shared_feat: torch.Tensor) -> torch.Tensor:

        return X