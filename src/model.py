import torch
import torch.nn as nn
import torch.utils.data
from sklearn.metrics import mean_squared_error

from model_bases import TorchModelBase
from tabnet import TabNet


class TabNetRegressor(TorchModelBase):

    def __init__(self, n_steps: int = 3, **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_steps = n_steps
        self.loss = nn.MSELoss() if not 'loss' in kwargs else kwargs['loss']

    def build_dataset(self, X, y=None):
        self.n_d = X.shape[1]
        if y:
            return torch.utils.data.TensorDataset(X, y)
        return torch.utils.data.TensorDataset(X)

    def build_graph(self):
        return TabNet(batch_dim=self.batch_size, n_d=self.n_d, n_steps=self.n_steps)

    def forward_propagate(self, X_batch: torch.Tensor, y_batch: torch.Tensor):
        batch_preds, self.agg_mask, reg = self.model(*X_batch)
        err = self.loss(batch_preds, y_batch)
        err += reg
        return err
    
    def score(self, X, y, device=None):
        preds = self.predict(X, device=device)
        return mean_squared_error(y.to_numpy(), preds.to_numpy())



from sklearn.datasets import load_boston
X, y = load_boston(return_X_y=True)

tn = TabNetRegressor(batch_size=16)
tn.fit(X, y)
print('Score: ', tn.score(X, y))

print('Attributes: ', tn.attributes)