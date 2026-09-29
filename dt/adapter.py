"""Adapters: framework-specific logic lives here, never in the graph executor.

Each adapter wraps ONE underlying model object and exposes a uniform
interface::

    fit(X: np.ndarray, y: np.ndarray) -> self
    predict(X: np.ndarray) -> np.ndarray
    is_fitted -> bool
    output_dim_ -> int | None   (columns produced; set after fit)
    save(dir) / load(dir)

Honesty rule: adapters never fake differentiability. Tree adapters expose
no gradients; TorchAdapter is the only one with an optimizer, and it trains
only on its own local inputs (materialized upstream outputs) -- gradients
never cross a tree boundary.
"""

from __future__ import annotations

import os
import pickle
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

from .data import to_numpy, ensure_2d, target_to_numpy
from .exceptions import DTError, UnfittedNodeError, SerializationError
from .schema import OutputKind, TaskType


class BaseAdapter(ABC):
    kind: str = "base"

    def __init__(self):
        self._fitted: bool = False
        self.output_dim_: Optional[int] = None

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseAdapter":
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def check_fitted(self, name: str = "") -> None:
        if not self._fitted:
            raise UnfittedNodeError(
                f"Node '{name}' is not fitted yet. Call graph.fit(X, y) before "
                f"graph.predict(X). (All trainable nodes in the DAG are fitted "
                f"sequentially in topological order.)"
            )

    @abstractmethod
    def save(self, directory: str) -> None:
        ...

    @abstractmethod
    def load(self, directory: str) -> None:
        ...

    def config(self) -> dict:
        return {"kind": self.kind}


# ---------------------------------------------------------------- sklearn ---
class SklearnAdapter(BaseAdapter):
    kind = "sklearn"

    def __init__(self, estimator: Any, *, output: str = "prediction"):
        """``output``: 'prediction' (default), 'proba', or 'transform'.

        - prediction -> estimator.predict -> (n,) or (n, k)
        - proba      -> estimator.predict_proba -> (n, k)
        - transform  -> estimator.transform -> (n, k)
        """
        super().__init__()
        if output not in ("prediction", "proba", "transform"):
            raise DTError(f"SklearnAdapter output must be prediction|proba|transform, got '{output}'.")
        self.estimator = estimator
        self.output = output

    def fit(self, X, y):
        X = ensure_2d(to_numpy(X), name="sklearn.fit input")
        y = target_to_numpy(y)
        if X.shape[0] != y.shape[0]:
            raise DTError(f"sklearn.fit: X rows {X.shape[0]} != y rows {y.shape[0]}.")
        self.estimator.fit(X, y)
        self._fitted = True
        # Probe output dim on a small slice.
        try:
            probe = self.predict(X[: min(5, len(X))])
            p = np.asanyarray(probe)
            self.output_dim_ = 1 if p.ndim == 1 else int(p.shape[1])
        except Exception:
            self.output_dim_ = None
        return self

    def predict(self, X):
        X = ensure_2d(to_numpy(X), name="sklearn.predict input")
        if self.output == "proba":
            if not hasattr(self.estimator, "predict_proba"):
                raise DTError(f"Estimator {type(self.estimator).__name__} has no predict_proba.")
            return np.asanyarray(self.estimator.predict_proba(X))
        if self.output == "transform":
            # TRANSFORM kind = (n, k) feature matrix. Prefer .transform()
            # (PCA, scalers); fall back to .predict() for estimator banks
            # (e.g. MultiOutputRegressor) whose predictions ARE the features.
            if hasattr(self.estimator, "transform"):
                try:
                    return np.asanyarray(self.estimator.transform(X))
                except Exception:
                    pass
            if hasattr(self.estimator, "predict"):
                return np.asanyarray(self.estimator.predict(X))
            raise DTError(f"Estimator {type(self.estimator).__name__} has neither transform nor predict.")
        if not hasattr(self.estimator, "predict"):
            raise DTError(f"Estimator {type(self.estimator).__name__} has no predict.")
        return np.asanyarray(self.estimator.predict(X))

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "estimator.pkl"), "wb") as f:
            pickle.dump({"estimator": self.estimator, "output": self.output}, f)

    def load(self, directory: str) -> None:
        try:
            with open(os.path.join(directory, "estimator.pkl"), "rb") as f:
                obj = pickle.load(f)
        except Exception as e:
            raise SerializationError(f"SklearnAdapter load failed: {e}") from e
        self.estimator = obj["estimator"]
        self.output = obj.get("output", "prediction")
        self._fitted = True

    def config(self) -> dict:
        return {"kind": self.kind, "estimator": type(self.estimator).__name__, "output": self.output}


# ---------------------------------------------------------------- CatBoost --
class CatBoostAdapter(BaseAdapter):
    kind = "catboost"

    def __init__(self, estimator: Any, *, output: str = "prediction"):
        """Wraps a ``CatBoostRegressor``/``CatBoostClassifier`` instance.

        ``output``: 'prediction' -> .predict() ``(n,)``;
                    'proba'      -> .predict_proba() ``(n, k)``.
        NOTE: v0.1 converts pandas->numpy at the boundary, so only numeric
        data is supported (categorical passthrough is roadmap).
        """
        super().__init__()
        if output not in ("prediction", "proba"):
            raise DTError(f"CatBoostAdapter output must be prediction|proba, got '{output}'.")
        self.estimator = estimator
        self.output = output

    def _require_catboost(self) -> None:
        try:
            import catboost  # noqa: F401
        except ImportError as e:
            raise DTError("CatBoost is not installed. pip install catboost.") from e

    def fit(self, X, y):
        self._require_catboost()
        X = ensure_2d(to_numpy(X), name="catboost.fit input")
        y = target_to_numpy(y)
        if X.shape[0] != y.shape[0]:
            raise DTError(f"catboost.fit: X rows {X.shape[0]} != y rows {y.shape[0]}.")
        self.estimator.fit(X, y, verbose=False)
        self._fitted = True
        probe = self.predict(X[: min(5, len(X))])
        p = np.asanyarray(probe)
        self.output_dim_ = 1 if p.ndim == 1 else int(p.shape[1])
        return self

    def predict(self, X):
        X = ensure_2d(to_numpy(X), name="catboost.predict input")
        if self.output == "proba":
            if not hasattr(self.estimator, "predict_proba"):
                raise DTError("This CatBoost estimator has no predict_proba (use a Classifier).")
            return np.asanyarray(self.estimator.predict_proba(X))
        return np.asanyarray(self.estimator.predict(X))

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        self.estimator.save_model(os.path.join(directory, "model.cbm"))
        with open(os.path.join(directory, "meta.pkl"), "wb") as f:
            pickle.dump(
                {"class": type(self.estimator).__name__, "output": self.output,
                 "params": self.estimator.get_params()},
                f,
            )

    def load(self, directory: str) -> None:
        self._require_catboost()
        try:
            with open(os.path.join(directory, "meta.pkl"), "rb") as f:
                meta = pickle.load(f)
            from catboost import CatBoostRegressor, CatBoostClassifier

            cls = CatBoostClassifier if meta["class"] == "CatBoostClassifier" else CatBoostRegressor
            self.estimator = cls()
            self.estimator.load_model(os.path.join(directory, "model.cbm"))
            self.output = meta.get("output", "prediction")
            self._fitted = True
        except Exception as e:
            raise SerializationError(f"CatBoostAdapter load failed: {e}") from e

    def config(self) -> dict:
        return {"kind": self.kind, "estimator": type(self.estimator).__name__, "output": self.output}


# ----------------------------------------------------------------- XGBoost --
class XGBoostAdapter(BaseAdapter):
    kind = "xgboost"

    def __init__(self, estimator: Any, *, output: str = "prediction"):
        """Wraps an sklearn-API ``XGBRegressor``/``XGBClassifier``.

        ``output``: 'prediction' -> .predict(); 'proba' -> .predict_proba().
        (Raw ``xgboost.Booster`` objects are not supported in v0.1.)
        """
        super().__init__()
        if output not in ("prediction", "proba"):
            raise DTError(f"XGBoostAdapter output must be prediction|proba, got '{output}'.")
        self.estimator = estimator
        self.output = output

    def _require_xgb(self) -> None:
        try:
            import xgboost  # noqa: F401
        except ImportError as e:
            raise DTError("XGBoost is not installed. pip install xgboost.") from e

    def fit(self, X, y):
        self._require_xgb()
        X = ensure_2d(to_numpy(X), name="xgboost.fit input")
        y = target_to_numpy(y)
        if X.shape[0] != y.shape[0]:
            raise DTError(f"xgboost.fit: X rows {X.shape[0]} != y rows {y.shape[0]}.")
        self.estimator.fit(X, y)
        self._fitted = True
        probe = self.predict(X[: min(5, len(X))])
        p = np.asanyarray(probe)
        self.output_dim_ = 1 if p.ndim == 1 else int(p.shape[1])
        return self

    def predict(self, X):
        X = ensure_2d(to_numpy(X), name="xgboost.predict input")
        if self.output == "proba":
            if not hasattr(self.estimator, "predict_proba"):
                raise DTError("This XGBoost estimator has no predict_proba (use a Classifier).")
            return np.asanyarray(self.estimator.predict_proba(X))
        return np.asanyarray(self.estimator.predict(X))

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        self.estimator.save_model(os.path.join(directory, "model.ubj"))
        with open(os.path.join(directory, "meta.pkl"), "wb") as f:
            pickle.dump(
                {"class": type(self.estimator).__name__, "output": self.output,
                 "params": self.estimator.get_params()},
                f,
            )

    def load(self, directory: str) -> None:
        self._require_xgb()
        try:
            with open(os.path.join(directory, "meta.pkl"), "rb") as f:
                meta = pickle.load(f)
            from xgboost import XGBRegressor, XGBClassifier

            cls = XGBClassifier if meta["class"] == "XGBClassifier" else XGBRegressor
            self.estimator = cls()
            self.estimator.load_model(os.path.join(directory, "model.ubj"))
            self.output = meta.get("output", "prediction")
            self._fitted = True
        except Exception as e:
            raise SerializationError(f"XGBoostAdapter load failed: {e}") from e

    def config(self) -> dict:
        return {"kind": self.kind, "estimator": type(self.estimator).__name__, "output": self.output}


# ------------------------------------------------------------------- Torch --
class TorchAdapter(BaseAdapter):
    kind = "torch"

    def __init__(
        self,
        module: Any,
        *,
        loss_fn: Any = None,
        optimizer_fn: Any = None,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: Optional[str] = None,
    ):
        """Wraps a ``torch.nn.Module`` for LOCAL supervised training.

        The module is trained with plain mini-batch SGD on this node's own
        materialized inputs ``X`` and the graph target ``y``. No gradient
        flows into/out of this node from anywhere else in the graph.
        """
        super().__init__()
        self.module = module
        self.loss_fn = loss_fn
        self.optimizer_fn = optimizer_fn
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self._device = device
        self._in_features: Optional[int] = None
        self._out_features: Optional[int] = None

    @property
    def device(self) -> str:
        if self._device:
            return self._device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def fit(self, X, y):
        import torch

        from .data import to_tensor

        Xn = ensure_2d(to_numpy(X), name="torch.fit input")
        yn = target_to_numpy(y)
        if Xn.shape[0] != yn.shape[0]:
            raise DTError(f"torch.fit: X rows {Xn.shape[0]} != y rows {yn.shape[0]}.")
        dev = self.device
        self.module.to(dev)
        self.module.train()
        Xt = to_tensor(Xn).to(dev)
        # Target tensor: (n,) or (n, k) float; classification-long support
        # is left to the user's loss_fn (we pass float and let MSELoss-style
        # defaults work; CrossEntropy users should shape targets accordingly
        # via a FunctionNode -- documented in README).
        if yn.ndim == 1:
            yt = torch.from_numpy(np.ascontiguousarray(yn.astype(np.float32))).to(dev)
        else:
            yt = to_tensor(yn.astype(np.float32)).to(dev)

        loss_fn = self.loss_fn or torch.nn.MSELoss()
        if self.optimizer_fn is not None:
            opt = self.optimizer_fn(self.module.parameters())
        else:
            opt = torch.optim.Adam(self.module.parameters(), lr=self.lr)
        n = Xt.shape[0]
        bs = max(1, min(self.batch_size, n))
        for _ in range(max(1, self.epochs)):
            perm = torch.randperm(n)
            for s in range(0, n, bs):
                idx = perm[s : s + bs]
                xb, yb = Xt[idx], yt[idx]
                opt.zero_grad()
                out = self.module(xb)
                # Align shapes for scalar losses: model (b,1) vs target (b,)
                if out.dim() == 2 and out.shape[1] == 1 and yb.dim() == 1:
                    out_l = out.squeeze(1)
                else:
                    out_l = out
                    if yb.dim() == 1 and out.dim() == 2:
                        yb = yb.unsqueeze(1).expand_as(out)
                loss = loss_fn(out_l, yb)
                loss.backward()
                opt.step()
        self._fitted = True
        self._in_features = int(Xn.shape[1])
        with torch.no_grad():
            self.module.eval()
            probe = self.module(Xt[: min(5, n)])
            p = probe.detach().cpu().numpy()
            self._out_features = 1 if p.ndim == 1 else int(p.shape[1])
            self.output_dim_ = self._out_features
        return self

    def predict(self, X):
        import torch

        from .data import to_tensor

        Xn = ensure_2d(to_numpy(X), name="torch.predict input")
        self.module.to(self.device)
        self.module.eval()
        with torch.no_grad():
            out = self.module(to_tensor(Xn).to(self.device))
            return out.detach().cpu().numpy()

    def save(self, directory: str) -> None:
        import torch

        os.makedirs(directory, exist_ok=True)
        torch.save(self.module.state_dict(), os.path.join(directory, "module_state.pt"))
        with open(os.path.join(directory, "meta.pkl"), "wb") as f:
            pickle.dump(
                {
                    "module_class": type(self.module).__module__ + "." + type(self.module).__qualname__,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "lr": self.lr,
                    "in_features": self._in_features,
                    "out_features": self._out_features,
                },
                f,
            )

    def load(self, directory: str) -> None:
        import torch

        # Defaults for instances created via __new__ (deserialization path).
        if not hasattr(self, "epochs"):
            self.epochs = 10
        if not hasattr(self, "batch_size"):
            self.batch_size = 32
        if not hasattr(self, "lr"):
            self.lr = 1e-3
        for attr in ("loss_fn", "optimizer_fn", "_device", "_in_features", "_out_features"):
            if not hasattr(self, attr):
                setattr(self, attr, None)
        try:
            with open(os.path.join(directory, "meta.pkl"), "rb") as f:
                meta = pickle.load(f)
            # Reconstruct the module via its zero-arg constructor, then load
            # state. Modules needing constructor args must be re-registered
            # by user code before load (documented limitation).
            mod_cls = _import_qualname(meta["module_class"])
            try:
                module = mod_cls()
            except TypeError as e:
                raise SerializationError(
                    f"TorchAdapter reload needs a zero-arg constructor for "
                    f"{meta['module_class']}; got: {e}. Re-create the graph in "
                    f"code (same architecture) and call graph.load_nodes() or "
                    f"use a zero-arg module."
                ) from e
            state = torch.load(os.path.join(directory, "module_state.pt"), map_location="cpu")
            module.load_state_dict(state)
            self.module = module
            self.epochs = meta.get("epochs", self.epochs)
            self.batch_size = meta.get("batch_size", self.batch_size)
            self.lr = meta.get("lr", self.lr)
            self._in_features = meta.get("in_features")
            self._out_features = meta.get("out_features")
            self.output_dim_ = self._out_features
            self._fitted = True
        except SerializationError:
            raise
        except Exception as e:
            raise SerializationError(f"TorchAdapter load failed: {e}") from e

    def config(self) -> dict:
        return {
            "kind": self.kind,
            "module": type(self.module).__name__,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
        }


def _import_qualname(qn: str) -> Any:
    mod_name, _, attr = qn.rpartition(".")
    import importlib

    mod = importlib.import_module(mod_name)
    obj = mod
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


# --------------------------------------------------------------- LightGBM ---
# Planned (not in v0.1): same pattern as CatBoost/XGBoost. Importing here
# would add a hard dependency, so it intentionally does not exist yet.
ADAPTER_REGISTRY = {
    "sklearn": SklearnAdapter,
    "catboost": CatBoostAdapter,
    "xgboost": XGBoostAdapter,
    "torch": TorchAdapter,
}
