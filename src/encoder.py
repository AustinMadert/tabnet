from typing import List
import torch
import torch.nn as nn
from torch.nn.modules.container import Sequential

from transformers import FeatureBlock, AttentiveTransformer


class TabNetEncoderStep(nn.Module):

    def __init__(self, shared_feat: FeatureBlock, input_dim: int, batch_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.shared_feat = shared_feat
        self.input_dim = input_dim
        self.batch_dim = batch_dim
        self.hidden_dim = hidden_dim
        self.att = AttentiveTransformer(input_dim=input_dim)
        self.feat = self.build_feature_transformer()
        self.p = torch.ones(self.batch_dim, self.input_dim)
        self.relu = nn.ReLU()


    def build_feature_transformer(self) -> nn.Module:
        return nn.Sequential(
            self.shared_feat,
            FeatureBlock(self.hidden_dim)
        )


    def forward(self, X: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        mask, self.p = self.att(a, self.p)
        X *= mask
        out = self.feat(X)
        # Split activations from outputs in the columns dimension
        splits = (int(out.shape[1] / 2), int(out.shape[1] / 2))
        out, a = out.split(splits, dim=1)
        out = self.relu(out)
        return out, a, mask



class TabNetEncoder(nn.Module):

    def __init__(self, shared_feat: nn.Sequential, input_dim: int, batch_dim: int,  
                output_dim: int = None, hidden_dim: int = None, n_steps: int = 10) -> None:
        super().__init__()
        self.shared_feat = shared_feat
        self.n_steps = n_steps
        self.batch_dim = batch_dim
        self.input_dim = input_dim
        self.output_dim = input_dim if not output_dim else output_dim
        self.hidden_dim = hidden_dim
        self.bn = nn.BatchNorm1d(input_dim)
        self.fc = nn.Linear(input_dim, input_dim)
        self.steps = self.build_encoder_steps()
        self.attributes = torch.zeros(self.batch_dim, self.input_dim)
        self.output = torch.zeros(self.batch_dim, self.input_dim)


    def build_encoder_steps(self) -> List[nn.Module]:
        return [TabNetEncoderStep(self.shared_feat, self.input_dim, self.batch_dim, self.hidden_dim) 
                for _ in range(self.n_steps)]


    def forward(self, X: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        for step in self.steps:
            out, a, mask = step(X, a)
            self.output += out
            self.attributes += (mask * out)

        self.output = self.fc(self.output)

        return self.output, self.attributes