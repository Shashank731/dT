"""Torch chain example (module-level classes so .dt reload can re-import them)."""

import torch
from sklearn.linear_model import LinearRegression

from dt import SklearnNode, TorchNode, dT


class SmallMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(12, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x)


class TorchChain(dT):
    def __init__(self, epochs=5):
        super().__init__()
        self.t = TorchNode(SmallMLP(), in_features=12, epochs=epochs)
        self.head = SklearnNode(LinearRegression(), in_features=1)

    def forward(self, x):
        return self.head(self.t(x))
