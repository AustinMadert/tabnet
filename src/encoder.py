from typing import List
import torch
import torch.nn as nn
from torch.nn.modules.container import Sequential

from transformers import FeatureBlock, AttentiveTransformer


class TabNetEncoderStep(nn.Module):

    def __init__(self, shared_feat: FeatureBlock, input_size: int) -> None:
        super().__init__()
        self.shared_feat = shared_feat
        self.input_size = input_size
        self.att = AttentiveTransformer(input_size=input_size)
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
        # Split activations from outputs in the columns dimension
        out, a = X.split(X.shape[1] / 2, dim=1)
        out = nn.ReLU(out)
        return out, a, mask



class TabNetEncoder(nn.Module):

    def __init__(self, shared_feat: FeatureBlock, input_size: int, batch_size: int,  
                output_size: int = None, n_steps: int = 10) -> None:
        super().__init__()
        self.shared_feat = shared_feat
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.input_size = input_size * 2
        if not output_size:
            self.output_size = input_size
        self.feat = self.build_feature_transformer()
        self.bn = nn.BatchNorm2d(input_size)
        self.fc = nn.Linear(input_size, input_size)
        self.steps = self.build_encoder_steps()
        self.attributes = torch.zeros(self.batch_size, self.input_size)
        self.output = torch.zeros(self.batch_size, self.input_size)
        

    def build_feature_transformer(self) -> nn.Module:
        return nn.Sequential(
            self.shared_feat,
            FeatureBlock(self.input_size)
        )

    def build_encoder_steps(self) -> List[nn.Module]:
        return [TabNetEncoderStep(self.shared_feat, self.input_size) 
                for _ in range(self.n_steps)]


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.bn(X)
        X = self.feat(X)
        _, a = X.split(X.shape[1] / 2, dim=1)

        for step in self.steps:
            out, a, mask = step(X, a)
            self.output += out
            self.attributes += (mask * out)

        self.output = self.fc(self.output)

        return self.output, self.attributes