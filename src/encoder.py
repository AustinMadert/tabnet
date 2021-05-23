import torch
import torch.nn as nn

from transformers import FeatureBlock, AttentiveTransformer


class TabNetStep(nn.Module):

    def __init__(self, input_size: int, shared_feat_transformer: FeatureBlock) -> None:
        super().__init__()
        self.att = AttentiveTransformer(input_size=input_size)
        self.shared_feat = shared_feat_transformer
        self.feat = self.build_feature_transformer()
        self.p = torch.ones(self.input_size, self.input_size)


    def build_feature_transformer(self) -> nn.Module:
        return nn.Sequential(
            self.shared_feat,
            FeatureBlock(self.input_size)
        )


    def forward(self, X: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        mask, self.p = self.att(a, self.p)
        X *= mask
        X = self.feat(X)
        # TODO: figure out current split operation
        out, a = torch.split(X)
        out = nn.ReLU(out)
        return out, a, mask



class TabNetEncoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X