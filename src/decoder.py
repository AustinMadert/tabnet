from typing import List
import torch
import torch.nn as nn

from transformers import FeatureBlock


class TabNetDecoderStep(nn.Module):

    def __init__(self, shared_feat: FeatureBlock, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim if not output_dim else output_dim
        self.feat = self.build_feature_transformer(shared_feat)
        self.fc = nn.Linear(hidden_dim, self.output_dim)


    def build_feature_transformer(self, shared_feat: FeatureBlock) -> nn.Module:
        return nn.Sequential(
            shared_feat,
            FeatureBlock(self.hidden_dim)
        )


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.feat(X)
        features = self.fc(X)
        return features


class TabNetDecoder(nn.Module):

    def __init__(self, shared_feat: FeatureBlock, input_dim: int, output_dim: int, 
                batch_dim: int, hidden_dim: int, n_steps: int)  -> None:
        super().__init__()
        self.shared_feat = shared_feat
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_steps = n_steps
        self.batch_dim = batch_dim
        self.output = torch.zeros(batch_dim, input_dim)
        self.steps = self.build_decoder_steps()


    def build_decoder_steps(self) -> List[nn.Module]:
        return [TabNetDecoderStep(self.shared_feat, self.hidden_dim, self.output_dim)
                for _ in range(self.n_steps)]


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        for step in self.steps:
            out = step(X)
            self.output += out
        return self.output
