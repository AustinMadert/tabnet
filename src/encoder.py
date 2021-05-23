import torch
import torch.nn as nn

from transformers import FeatureBlock, AttentiveTransformer


class TabNetStep(nn.Module):

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.att = AttentiveTransformer(input_size=input_size)
        self.feat = self.build_feature_transformer()
        self.p = self.build_p()


    def build_p(self) -> torch.Tensor:
        pass


    def build_feature_transformer(self) -> nn.Module:
        pass


    def forward(self, X: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        a = self.att(a)
        return X



class TabNetEncoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X