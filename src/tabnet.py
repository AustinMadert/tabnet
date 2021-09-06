import torch
import torch.nn as nn
from decoder import TabNetDecoder
from encoder import TabNetEncoder
from transformers import FeatureBlock
from activations import GLU
from normalization import GhostBatchNormalization


class TabNet(nn.Module):

    def __init__(self, batch_dim: int, n_d: int, n_steps: int = 5, 
                 output_dim: int = None, n_a: int = None) -> None:
        super().__init__()
        n_a = n_d if not n_a else n_a
        if not output_dim:
            output_dim = n_d
        hidden_dim = n_d + n_a
        self.splits = (n_d, n_a)
        self.shared_feat = nn.Sequential(
            nn.Linear(n_d, hidden_dim),
            GhostBatchNormalization(hidden_dim, 8),
            GLU(n_d=hidden_dim),
            FeatureBlock(hidden_dim, num_sub_blocks=1)
        )
        self.feat = nn.Sequential(
            self.shared_feat,
            FeatureBlock(hidden_dim)
        )
        kwargs = {
            'shared_feat': self.shared_feat,
            'n_d': n_d,
            'batch_dim': batch_dim,
            'hidden_dim': hidden_dim,
            'n_steps': n_steps
        }
        self.decoder = TabNetDecoder(**kwargs)
        kwargs['output_dim'] = output_dim
        self.encoder = TabNetEncoder(**kwargs)
        self.bn = GhostBatchNormalization(n_d, 8)



    # TODO: need to add flags to control mode of operation, or build separate classes
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.bn(X)
        out = self.shared_feat(X)
        _, a = out.split(self.splits, dim=1)
        encoded, agg_mask, reg = self.encoder(X, a)
        decoded = self.decoder(encoded)
        return decoded, agg_mask, reg
