import torch
import torch.nn as nn

from transformers import FeatureBlock


class TabNetDecoderStep(nn.Module):

    def __init__(self) -> None:
        super().__init__()


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        pass


class TabNetDecoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()


    def build_feature_transformer(self, shared_feat: FeatureBlock) -> nn.Module:
        return nn.Sequential(
            shared_feat,
            FeatureBlock(self.input_size)
        )


    def forward(self, X: torch.Tensor, shared_feat: torch.Tensor) -> torch.Tensor:
        feat_transformer = self.build_feature_transformer(shared_feat)

        return X