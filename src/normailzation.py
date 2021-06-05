import torch
import torch.nn as nn
import math


class GhostBatchNormalization(nn.Module):

    
    def __init__(self, num_features: int, virtual_batch_size: int) -> None:
        super().__init__()
        self.vbs = virtual_batch_size
        self.bn = nn.BatchNorm1d(num_features=num_features)

    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        num_batches = math.floor(X.shape[0] / self.vbs)
        for i in range(num_batches):
            lower = self.vbs * i
            X[lower: self.vbs + lower, :] = self.bn(X[lower: self.vbs + lower, :])
        return X
        
