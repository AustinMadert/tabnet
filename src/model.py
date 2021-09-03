import torch
import torch.nn as nn
import torch.utils.data
from sklearn.

from model_bases import TorchModelBase
from tabnet import TabNet


class TabNetRegressor(TorchModelBase):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def build_dataset(self, X, y=None):
        self.n_d = X.shape[1]
        if y:
            return torch.utils.data.TensorDataset(X, y)
        return torch.utils.data.TensorDataset(X)

    def build_graph(self):
        return TabNet(batch_dim=self.batch_size, n_d=self.n_d)
    
    def score(self, X, y, device=None):
        preds = self.predict(X, device=device)
        return 

