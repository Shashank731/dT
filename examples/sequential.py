"""Sequential example: CatBoost -> XGBoost.  Input(12) -> CB -> XGB -> Output."""

import numpy as np
from catboost import CatBoostRegressor
from sklearn.datasets import make_regression
from xgboost import XGBRegressor

from dt import CatBoostNode, XGBoostNode, dT


def make_data(n=200):
    return make_regression(n_samples=n, n_features=12, noise=5.0, random_state=0)


class SequentialModel(dT):
    def __init__(self):
        super().__init__()
        self.cb = CatBoostNode(
            CatBoostRegressor(iterations=50, verbose=False), in_features=12
        )
        self.xgb = XGBoostNode(XGBRegressor(n_estimators=50), in_features=1)

    def forward(self, x):
        return self.xgb(self.cb(x))


if __name__ == "__main__":
    X, y = make_data()
    m = SequentialModel()
    m.fit(X, y)
    print(m.describe())
    print("pred:", m.predict(X)[:5])
