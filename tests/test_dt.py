import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression

from dt import (
    CatBoostNode,
    ConcatMerge,
    FunctionNode,
    Graph,
    MeanMerge,
    SklearnNode,
    SumMerge,
    TorchNode,
    XGBoostNode,
    dT,
)
from dt.exceptions import (
    DataError,
    ShapeMismatchError,
    UnfittedNodeError,
)


@pytest.fixture(scope="module")
def reg():
    return make_regression(n_samples=80, n_features=12, noise=5.0, random_state=0)


@pytest.fixture(scope="module")
def clf():
    return make_classification(
        n_samples=90, n_features=12, n_informative=6, n_classes=2, random_state=0
    )


def _cb_reg(**kw):
    from catboost import CatBoostRegressor

    kw.setdefault("iterations", 8)
    kw.setdefault("verbose", False)
    return CatBoostRegressor(**kw)


def _xgb_reg(**kw):
    from xgboost import XGBRegressor

    kw.setdefault("n_estimators", 8)
    return XGBRegressor(**kw)


# -- sequential composition: 12 -> 4 -> 1 -------------------------------
def test_sequential_12_to_4_to_1(reg):
    """Stacking with multi-output intermediate: 12 -> 4 -> 1.

    '4' here is explicit: 4 estimators' predictions become 4 features
    (TRANSFORM kind), not a Linear(12, 4) projection.
    """
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.tree import DecisionTreeRegressor

    X, y = reg
    Y4 = np.stack([y, y + 1.0, y + 2.0, y + 3.0], axis=1)  # device to get 4 cols

    class Seq(dT):
        def __init__(self):
            super().__init__()
            self.bank = SklearnNode(
                MultiOutputRegressor(DecisionTreeRegressor(max_depth=3)),
                in_features=12,
                output="transform",
                out_dim=4,
            )
            self.head = SklearnNode(LinearRegression(), in_features=4)

        def forward(self, x):
            return self.head(self.bank(x))

    # Fit bank directly on 4-dim target to prove the contract, then head.
    m = Seq()
    m.bank.fit(X, Y4)
    head_X = m.bank.predict(X)
    assert head_X.shape == (len(X), 4)
    m.head.fit(head_X, y)
    pred = m.predict(X)
    assert pred.shape == (len(X),)


def test_sequential_catboost_xgboost(reg):
    X, y = reg

    class MyModel(dT):
        def __init__(self):
            super().__init__()
            self.cb = CatBoostNode(_cb_reg(), in_features=12)
            self.xgb = XGBoostNode(_xgb_reg(), in_features=1)

        def forward(self, x):
            return self.xgb(self.cb(x))

    m = MyModel()
    m.fit(X, y)
    pred = m.predict(X)
    assert pred.shape == (len(X),)
    # __call__ == predict (inference API)
    np.testing.assert_allclose(m(X), pred)


# -- invalid contracts --------------------------------------------------
def test_invalid_width_raises_clear_error(reg):
    X, y = reg

    class Bad(dT):
        def __init__(self):
            super().__init__()
            self.cb = CatBoostNode(_cb_reg(), in_features=4)

        def forward(self, x):
            return self.cb(x)

    with pytest.raises(ShapeMismatchError, match="expects 4 features, but got 12"):
        Bad().fit(X, y)


def test_proba_dim_mismatch(clf):
    X, y = clf

    class BadProba(dT):
        def __init__(self):
            super().__init__()
            from catboost import CatBoostClassifier

            # binary problem -> proba dim is 2, but node declares 7
            self.cb = CatBoostNode(
                CatBoostClassifier(iterations=8, verbose=False),
                in_features=12, output="proba", out_dim=7)

        def forward(self, x):
            return self.cb(x)

    with pytest.raises(ShapeMismatchError, match="dim=7"):
        BadProba().fit(X, y)


def test_symbolic_mismatch_caught_at_trace_time(reg):
    """12 -> proba(3) -> node expecting 4 must fail before any fitting."""
    X, y = reg

    class BadChain(dT):
        def __init__(self):
            super().__init__()
            from catboost import CatBoostClassifier

            Xc, yc = make_classification(n_samples=40, n_features=12, n_classes=3,
                                         n_informative=5, n_clusters_per_class=1, random_state=0)
            self._Xc, self._yc = Xc, yc
            self.cb = CatBoostNode(
                CatBoostClassifier(iterations=5, verbose=False),
                in_features=12, output="proba", out_dim=3)
            self.head = SklearnNode(LinearRegression(), in_features=4)

        def forward(self, x):
            return self.head(self.cb(x))

    m = BadChain()
    with pytest.raises(ShapeMismatchError, match="expects 4 features, but upstream node produces 3"):
        m.fit(m._Xc, m._yc)


# -- branching / merging --------------------------------------------------
def test_branching_diamond_merge(reg):
    X, y = reg

    class Diamond(dT):
        def __init__(self):
            super().__init__()
            self.cb = CatBoostNode(_cb_reg(), in_features=12)
            self.xgb = XGBoostNode(_xgb_reg(), in_features=12)
            self.merge = ConcatMerge()
            self.head = SklearnNode(LinearRegression(), in_features=2)

        def forward(self, x):
            a = self.cb(x)
            b = self.xgb(x)
            return self.head(self.merge(a, b))

    m = Diamond()
    m.fit(X, y)
    assert m.predict(X).shape == (len(X),)


def test_merge_row_mismatch(reg):
    from dt.node import ConcatMerge as C

    m = C()
    with pytest.raises(DataError, match="row-count mismatch"):
        m._eager_call(np.zeros((10, 2)), np.zeros((7, 2)))


def test_sum_mean_merge_shapes():
    a = np.ones((5, 3))
    b = np.full((5, 3), 2.0)
    s = SumMerge()
    assert s._eager_call(a, b).shape == (5, 3)
    np.testing.assert_allclose(s._eager_call(a, b), 3.0)
    me = MeanMerge()
    np.testing.assert_allclose(me._eager_call(a, b), 1.5)


def test_shared_node_executes_once(reg):
    """Diamond reuse: the shared upstream node must run exactly once."""
    X, y = reg
    calls = {"n": 0}
    base = LinearRegression()

    orig_predict = base.predict

    class Counting(dT):
        def __init__(self):
            super().__init__()
            self.a = SklearnNode(LinearRegression(), in_features=12)
            self.b = SklearnNode(LinearRegression(), in_features=1)
            self.c = SklearnNode(LinearRegression(), in_features=1)
            self.merge = ConcatMerge()
            self.head = SklearnNode(LinearRegression(), in_features=2)

        def forward(self, x):
            h = self.a(x)
            left = self.b(h)
            right = self.c(h)
            return self.head(self.merge(left, right))

    m = Counting()
    m.fit(X, y)
    # count predict calls on node a during one predict
    n0 = {"n": 0}
    orig = m.a.adapter.predict
    m.a.adapter.predict = lambda X_: (n0.__setitem__("n", n0["n"] + 1), orig(X_))[1]
    m.predict(X)
    assert n0["n"] == 1, f"shared node executed {n0['n']} times, expected 1"


# -- multiple inputs / outputs ---------------------------------------------
def test_multiple_inputs_outputs(clf):
    X, y = clf
    rng = np.random.RandomState(0)
    X2 = rng.randn(len(X), 5)

    class H(dT):
        def __init__(self):
            super().__init__()
            self.a = SklearnNode(LogisticRegression(max_iter=500), in_features=12)
            self.b = SklearnNode(LogisticRegression(max_iter=500), in_features=5)
            self.merge = ConcatMerge()

        def forward(self, x1, x2):
            pa = self.a(x1)
            pb = self.b(x2)
            return (pa, self.merge(pa, pb))

    h = H()
    h.fit({"x1": X, "x2": X2}, y)
    pa, both = h.predict({"x1": X, "x2": X2})
    assert pa.shape == (len(X),)
    assert both.shape == (len(X), 2)
    with pytest.raises(DataError, match="Missing graph inputs"):
        h.predict({"x1": X})


# -- PyTorch integration -----------------------------------------------------
def test_torch_node_sequential(reg):
    import torch

    X, y = reg
    yn = ((y - y.mean()) / y.std()).astype(np.float32)

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l = torch.nn.Linear(12, 1)

        def forward(self, x):
            return self.l(x)

    class G(dT):
        def __init__(self):
            super().__init__()
            self.t = TorchNode(M(), in_features=12, epochs=3)
            self.head = SklearnNode(LinearRegression(), in_features=1)

        def forward(self, x):
            return self.head(self.t(x))

    g = G()
    g.fit(X.astype(np.float32), yn)
    assert g.predict(X.astype(np.float32)).shape == (len(X),)


def test_torch_tensor_and_pandas_inputs(reg):
    import pandas as pd
    import torch

    X, y = reg

    class G(dT):
        def __init__(self):
            super().__init__()
            self.a = SklearnNode(LinearRegression(), in_features=12)

        def forward(self, x):
            return self.a(x)

    g = G()
    g.fit(pd.DataFrame(X), pd.Series(y))
    assert g.predict(torch.tensor(X, dtype=torch.float64)).shape == (len(X),)
    assert g.predict(pd.DataFrame(X)).shape == (len(X),)


# -- CatBoost / XGBoost integration -------------------------------------------
def test_catboost_proba_output(clf):
    from catboost import CatBoostClassifier

    X, y = clf

    class G(dT):
        def __init__(self):
            super().__init__()
            self.cb = CatBoostNode(
                CatBoostClassifier(iterations=8, verbose=False),
                in_features=12, output="proba", out_dim=2)

        def forward(self, x):
            return self.cb(x)

    g = G()
    g.fit(X, y)
    out = g.predict(X)
    assert out.shape == (len(X), 2)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-5)


def test_xgboost_proba_output(clf):
    from xgboost import XGBClassifier

    X, y = clf

    class G(dT):
        def __init__(self):
            super().__init__()
            self.xgb = XGBoostNode(
                XGBClassifier(n_estimators=8), in_features=12,
                output="proba", out_dim=2)

        def forward(self, x):
            return self.xgb(x)

    g = G()
    g.fit(X, y)
    assert g.predict(X).shape == (len(X), 2)


# -- data conversion -------------------------------------------------------------
def test_data_conversion_zero_copy():
    import torch
    from dt.data import to_numpy, to_tensor, ensure_2d, as_column

    a = np.arange(12, dtype=np.float32).reshape(4, 3)
    t = to_tensor(a)
    assert isinstance(t, torch.Tensor) and t.shape == (4, 3)
    back = to_numpy(t)
    assert back.shape == (4, 3)
    assert as_column(np.arange(4)).shape == (4, 1)
    with pytest.raises(DataError, match="expected 2D"):
        ensure_2d(np.arange(4), name="x")


# -- serialization ------------------------------------------------------------------
def _tiny_graph():
    class G(dT):
        def __init__(self):
            super().__init__()
            self.a = SklearnNode(LinearRegression(), in_features=12)
            self.b = SklearnNode(LinearRegression(), in_features=1)

        def forward(self, x):
            return self.b(self.a(x))

    return G


def test_serialization_sklearn_chain(reg, tmp_path):
    X, y = reg
    G = _tiny_graph()
    g = G()
    g.fit(X, y)
    before = g.predict(X)
    p = g.save(str(tmp_path / "m.dt"))
    g2 = Graph.load(p)
    np.testing.assert_allclose(g2.predict(X), before)


def test_serialization_catboost_xgboost_dag(reg, tmp_path):
    X, y = reg

    class Diamond(dT):
        def __init__(self):
            super().__init__()
            self.cb = CatBoostNode(_cb_reg(), in_features=12)
            self.xgb = XGBoostNode(_xgb_reg(), in_features=12)
            self.merge = ConcatMerge()
            self.head = SklearnNode(LinearRegression(), in_features=2)

        def forward(self, x):
            return self.head(self.merge(self.cb(x), self.xgb(x)))

    m = Diamond()
    m.fit(X, y)
    before = m.predict(X)
    m.save(str(tmp_path / "dag.dt"))
    m2 = Graph.load(str(tmp_path / "dag.dt"))
    np.testing.assert_allclose(m2.predict(X), before)


def test_serialization_torch(reg, tmp_path):
    import torch

    X, y = reg
    yn = ((y - y.mean()) / y.std()).astype(np.float32)

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l = torch.nn.Linear(12, 1)

        def forward(self, x):
            return self.l(x)

    # define at module level equivalent: use repo example module for reload
    import sys

    sys.path.insert(0, "/home/shashank/Projects/dT/examples")
    from torch_chain import TorchChain

    g = TorchChain(epochs=2)
    g.fit(X.astype(np.float32), yn)
    before = g.predict(X.astype(np.float32))
    g.save(str(tmp_path / "t.dt"))
    g2 = Graph.load(str(tmp_path / "t.dt"))
    np.testing.assert_allclose(g2.predict(X.astype(np.float32)), before, rtol=1e-5)


# -- unfitted / error messages ---------------------------------------------------------
def test_unfitted_nodes(reg):
    X, y = reg

    class G(dT):
        def __init__(self):
            super().__init__()
            self.a = SklearnNode(LinearRegression(), in_features=12)

        def forward(self, x):
            return self.a(x)

    with pytest.raises(UnfittedNodeError, match="not fitted.*graph.fit"):
        G().predict(X)


def test_function_node_custom_model(reg):
    X, y = reg

    class G(dT):
        def __init__(self):
            super().__init__()
            self.a = SklearnNode(LinearRegression(), in_features=12)
            self.proj = FunctionNode(lambda H: H[:, :1] * 2.0, in_features=1,
                                     out_dim=1, output="transform")
            self.head = SklearnNode(LinearRegression(), in_features=1)

        def forward(self, x):
            return self.head(self.proj(self.a(x)))

    g = G()
    g.fit(X, y)
    assert g.predict(X).shape == (len(X),)
