"""DAG example: diamond branching with a Concat merge.

            Input
           /     \\
     CatBoost   XGBoost
          \\         /
             Merge (concat -> 2 features)
               |
         LinearRegression head
               |
             Output
"""

from catboost import CatBoostRegressor
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from dt import CatBoostNode, ConcatMerge, SklearnNode, XGBoostNode, dT


class DiamondModel(dT):
    def __init__(self):
        super().__init__()
        self.cb = CatBoostNode(
            CatBoostRegressor(iterations=50, verbose=False), in_features=12
        )
        self.xgb = XGBoostNode(XGBRegressor(n_estimators=50), in_features=12)
        self.merge = ConcatMerge()
        self.head = SklearnNode(LinearRegression(), in_features=2)

    def forward(self, x):
        a = self.cb(x)
        b = self.xgb(x)
        return self.head(self.merge(a, b))


if __name__ == "__main__":
    X, y = make_regression(n_samples=200, n_features=12, noise=5.0, random_state=0)
    m = DiamondModel()
    m.fit(X, y)
    print(m.describe())
    print("pred:", m.predict(X)[:5])
