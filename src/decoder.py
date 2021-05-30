from typing import List
import torch
import torch.nn as nn

from transformers import FeatureBlock


class TabNetDecoderStep(nn.Module):

    def __init__(self, shared_feat: nn.Sequential, hidden_dim: int, n_d: int) -> None:
        super().__init__()
        self.feat = nn.Sequential(shared_feat, FeatureBlock(hidden_dim))
        self.fc = nn.Linear(hidden_dim, n_d)


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.feat(X)
        features = self.fc(X)
        return features


class TabNetDecoder(nn.Module):

    def __init__(self, shared_feat: FeatureBlock, n_d: int, batch_dim: int, 
                 hidden_dim: int, n_steps: int)  -> None:
        super().__init__()
        self.output = torch.zeros(batch_dim, n_d)
        self.steps = [TabNetDecoderStep(shared_feat, hidden_dim, n_d)
                      for _ in range(n_steps)]


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        for step in self.steps:
            out = step(X)
            self.output += out
        return self.output
