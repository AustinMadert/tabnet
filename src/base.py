from typing import TypeVar

import torch
import torch.nn as nn

ModelBase = TypeVar('ModelBase')


class TorchBase(object):

    def __init__(self) -> None:
        pass


    def build_dataset(self):
        return NotImplementedError


    def build_graph(self):
        return NotImplementedError


    def fit(self, X: torch.Tensor, y: torch.Tensor = None) -> ModelBase:
        return self


    def predict(self, X: torch.Tensor) -> torch.Tensor:
        pass