from typing import List
import torch
import torch.nn as nn

from transformers import FeatureBlock


class TabNetDecoderStep(nn.Module):

    def __init__(self, shared_feat: FeatureBlock, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        if not output_dim:
            self.output_dim = input_dim
        self.feat = self.build_feature_transformer(shared_feat)
        self.fc = nn.Linear(input_dim, self.output_dim)


    def build_feature_transformer(self, shared_feat: FeatureBlock) -> nn.Module:
        return nn.Sequential(
            shared_feat,
            FeatureBlock(self.input_dim)
        )


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.feat(X)
        features = self.fc(X)
        return features


class TabNetDecoder(nn.Module):

    def __init__(self, shared_feat: FeatureBlock, input_dim: int, output_dim: int, 
                batch_dim: int, n_steps: int)  -> None:
        super().__init__()
        self.shared_feat = shared_feat
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_steps = n_steps
        self.batch_dim = batch_dim
        self.output = torch.zeros(batch_dim, input_dim)
        self.steps = self.build_decoder_steps()


    def build_decoder_steps(self) -> List[nn.Module]:
        return [TabNetDecoderStep(self.shared_feat, self.input_dim, self.output_dim)
                for _ in range(self.n_steps)]


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        for step in self.steps:
            X = step(X)
            self.output += X
        return self.output
