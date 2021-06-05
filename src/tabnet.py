import torch
import torch.nn as nn
from decoder import TabNetDecoder
from encoder import TabNetEncoder
from transformers import FeatureBlock
from activations import GLU
from normailzation import GhostBatchNormalization


class TabNet(nn.Module):

    def __init__(self, batch_dim: int, n_d: int, n_steps: int = 5, 
                 output_dim: int = None, n_a: int = None) -> None:
        super().__init__()
        n_a = n_d if not n_a else n_a
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
        encoded, attributes = self.encoder(X, a)
        decoded = self.decoder(encoded)
        return decoded, attributes


from sklearn.datasets import load_boston
X, y = load_boston(return_X_y=True)

tn = TabNet(
    n_d=X.shape[1], 
    batch_dim=16, 
    n_steps=3,
    output_dim=X.shape[1],
    n_a=X.shape[1]
)
X = torch.Tensor(X[:16, :])
X, attributes = tn.forward(X)

print(X, attributes)