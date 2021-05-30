import torch
import torch.nn as nn
from decoder import TabNetDecoder
from encoder import TabNetEncoder
from transformers import FeatureBlock
from activations import GLU


class TabNet(nn.Module):

    def __init__(self, input_dim: int, batch_dim: int,  n_steps: int = 10, 
                output_dim: int = None, n_d: int = None, n_a: int = None) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.batch_dim = batch_dim
        self.n_steps = n_steps
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.hidden_dim = n_d + n_a if n_d and n_a else self.input_dim * 2
        self.shared_feat = self.build_shared_feature_transformer()
        self.feat = self.build_feature_transformer()
        self.encoder = TabNetEncoder(
            shared_feat=self.shared_feat,
            input_dim=self.input_dim,
            batch_dim=batch_dim,
            output_dim=output_dim,
            hidden_dim=self.hidden_dim,
            n_steps=n_steps
        )
        self.decoder = TabNetDecoder(
            shared_feat=self.shared_feat,
            input_dim=input_dim,
            batch_dim=batch_dim,
            output_dim=output_dim,
            hidden_dim=self.hidden_dim,
            n_steps=n_steps
        )
        self.bn = nn.BatchNorm1d(input_dim)


    def build_shared_feature_transformer(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            GLU(input_dim=self.hidden_dim),
            FeatureBlock(self.hidden_dim, num_sub_blocks=1)
        )

    def build_feature_transformer(self) -> nn.Module:
        return nn.Sequential(
            self.shared_feat,
            FeatureBlock(self.hidden_dim)
        )

    # TODO: need to add flags to control mode of operation, or build separate classes
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.bn(X)
        out = self.shared_feat(X)
        splits = (int(out.shape[1] / 2), int(out.shape[1] / 2))
        _, a = out.split(splits, dim=1)
        encoded, attributes = self.encoder(X, a)
        decoded = self.decoder(encoded)
        return decoded, attributes


from sklearn.datasets import load_boston
X, y = load_boston(return_X_y=True)

tn = TabNet(
    input_dim=X.shape[1], 
    batch_dim=16, 
    n_steps=3,
    output_dim=X.shape[1],
    n_d=X.shape[1],
    n_a=X.shape[1]
)
# print(tn)
X = torch.Tensor(X[:16, :])
l = nn.Linear(13, 26)
# print(l(X).shape, l(X))
X, attributes = tn.forward(X)

print(X, attributes)