import torch
import torch.nn as nn

from base import TorchBase
from tabnet import TabNet


class TabNetModel(TorchBase):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    
    def build_graph(self):
        return TabNet()