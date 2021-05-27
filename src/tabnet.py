import torch
import torch.nn as nn
from decoder import TabNetDecoder
from encoder import TabNetEncoder
from transformers import FeatureBlock


class TabNet(nn.Module):

    def __init__(self, input_dim: int, batch_dim: int,  n_steps: int = 10, 
                output_dim: int = None) -> None:
        super().__init__()
        self.hidden_dim = input_dim * 2
        self.batch_dim = batch_dim
        self.n_steps = n_steps
        self.output_dim = output_dim
        self.shared_feat = self.build_shared_feature_transformer(input_dim)
        self.encoder = TabNetEncoder(
            shared_feat=self.shared_feat,
            input_dim=input_dim,
            batch_dim=batch_dim,
            output_dim=output_dim,
            n_steps=n_steps
        )
        self.decoder = TabNetDecoder(
            shared_feat=self.shared_feat,
            input_dim=input_dim,
            batch_dim=batch_dim,
            output_dim=output_dim,
            n_steps=n_steps
        )
        self.bn = nn.BatchNorm1d(input_dim)


    def build_shared_feature_transformer(self, input_dim) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.GLU(),
            FeatureBlock(self.hidden_dim, sub_block_dim=1)
        )

    # TODO: need to add flags to control mode of operation, or build separate classes
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.bn(X)
        encoded, attributes = self.encoder(X)
        X = self.decoder(encoded)
        return X, attributes


from sklearn.datasets import load_boston
X, y = load_boston(return_X_y=True)

tn = TabNet(X.shape[1], 16, 3)
print(tn)
X = torch.Tensor(X[:16, :])
l = nn.Linear(13, 26)
print(l(X).shape, l(X))
X, attributes = tn.forward(X)

print(X, attributes)