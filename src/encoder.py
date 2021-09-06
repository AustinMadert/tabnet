from typing import List
import torch
import torch.nn as nn
from torch.nn.modules.container import Sequential

from transformers import FeatureBlock, AttentiveTransformer
from normalization import GhostBatchNormalization


class TabNetEncoderStep(nn.Module):

    def __init__(self, shared_feat: nn.Sequential, n_d: int, batch_dim: int, 
                 hidden_dim: int) -> None:
        super().__init__()
        self.splits = (n_d, hidden_dim - n_d)
        self.att = AttentiveTransformer(batch_dim, n_d)
        self.feat = nn.Sequential(shared_feat, FeatureBlock(hidden_dim))
        self.relu = nn.ReLU()


    def forward(self, X: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        mask = self.att(a)
        X *= mask
        out = self.feat(X)
        # Split activations from outputs in the columns dimension
        out, a = out.split(self.splits, dim=1)
        out = self.relu(out)
        return out, a, mask



class TabNetEncoder(nn.Module):

    def __init__(self, shared_feat: nn.Sequential, n_d: int, batch_dim: int,  
                output_dim: int, hidden_dim: int, n_steps: int) -> None:
        super().__init__()
        self.n_steps = n_steps
        self.batch_dim = batch_dim
        self.n_d = n_d
        self.bn = GhostBatchNormalization(n_d, 8)
        self.fc = nn.Linear(n_d, output_dim)
        self.steps = [TabNetEncoderStep(shared_feat, n_d, batch_dim, hidden_dim) 
                      for _ in range(n_steps)]
        self.importances = torch.Tensor()
        self.output = torch.zeros(batch_dim, n_d)
        self.reg = torch.Tensor([0.])


    def sparsity_regularization(self, mask: torch.Tensor, eta: float = 0.00001) -> float:
        reg = (-mask * torch.log(mask + eta)) / (self.n_steps * self.batch_dim)
        return torch.sum(reg)


    def build_aggregate_mask(self, batches: list, masks: list) -> torch.Tensor:
        batches = torch.cat(batches, dim=-1)
        masks = torch.cat(masks, dim=-1)
        nu = batches.sum(dim=1).repeat(1, self.n_d, 1)
        numer = torch.sum(nu * masks, dim=-1)
        denom = torch.sum(numer ** 2)
        return numer / denom


    def forward(self, X: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        masks = []
        outputs = []
        for step in self.steps:
            out, a, mask = step(X, a)
            self.reg += self.sparsity_regularization(mask)
            self.output += out
            masks.append(mask.unsqueeze(dim=-1))
            outputs.append(out.unsqueeze(dim=-1))
        agg_mask = self.build_aggregate_mask(outputs, masks)        

        self.output = self.fc(self.output)

        return self.output, agg_mask, self.reg