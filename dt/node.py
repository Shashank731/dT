"""Nodes: typed units of computation. All framework logic lives in adapters."""

from __future__ import annotations

import contextvars
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .adapter import BaseAdapter, CatBoostAdapter, SklearnAdapter, TorchAdapter, XGBoostAdapter
from .data import as_column, check_row_count, ensure_2d, to_numpy
from .exceptions import DTError, DataError, SchemaMismatchError, ShapeMismatchError, UnfittedNodeError
from .schema import FeatureSchema, OutputKind, OutputSchema, TaskType

# Active tracing graph (set while dT._trace_ runs forward with Symbolics).
_current_trace: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar("dt_trace", default=None)

_counter = [0]


def _auto_name(prefix: str) -> str:
    _counter[0] += 1
    return f"{prefix}{_counter[0]}"


class Symbolic:
    """Placeholder for a node's output during graph tracing.

    Carries the producer node + its declared output schema so connections can
    be validated symbolically, before any real data flows.
    """

    def __init__(self, node: "BaseNode", schema: OutputSchema, *, name: str = ""):
        self.node = node
        self.schema = schema
        self.name = name or f"sym_{node.name}"

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Symbolic({self.node.name} -> {self.schema.describe()})"


def _is_symbolic(x: Any) -> bool:
    return isinstance(x, Symbolic)


def _as_symbolic_list(args: Tuple) -> List[Symbolic]:
    flat: List[Symbolic] = []
    for a in args:
        if isinstance(a, (list, tuple)):
            for v in a:
                if not _is_symbolic(v):
                    raise DTError("Cannot mix Symbolic and real data in one Node call.")
                flat.append(v)
        elif _is_symbolic(a):
            flat.append(a)
        else:
            raise DTError("Cannot mix Symbolic and real data in one Node call.")
    return flat


class BaseNode:
    """Base class for all nodes.

    Subclasses declare ``input_schema`` (what widths/tasks they accept) and
    ``output_schema`` (what they promise to produce). ``__call__`` has two
    modes:

    - Trace mode (any arg is Symbolic): validate schemas symbolically,
      register ``(producer -> self)`` edges on the active trace, return a new
      Symbolic. No model code runs.
    - Eager mode (real data): validate real shapes, run the adapter, validate
      the produced output. Used by the executor and standalone node use.
    """

    adapter: Optional[BaseAdapter] = None
    kind: str = "base"

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        input_schema: Optional[FeatureSchema] = None,
        output_schema: Optional[OutputSchema] = None,
    ):
        self.name = name or _auto_name(self.kind + "_")
        self.input_schema = input_schema or FeatureSchema()
        self.output_schema = output_schema or OutputSchema(kind=OutputKind.PREDICTION, task=TaskType.UNSPECIFIED)

    # -- core protocol -------------------------------------------------
    def __call__(self, *args: Any) -> Any:
        if any(_is_symbolic(a) for a in args) or any(isinstance(a, (list, tuple)) and any(_is_symbolic(v) for v in a) for a in args):
            return self._trace_call(*args)
        return self._eager_call(*args)

    def _trace_call(self, *args: Any) -> Symbolic:
        syms = _as_symbolic_list(args)
        trace = _current_trace.get()
        if trace is None:
            raise DTError(
                f"Node '{self.name}' received Symbolic inputs outside a graph trace. "
                f"Symbolics are only created inside dT.forward during fit/predict/save."
            )
        for s in syms:
            self._validate_symbolic(s.schema)
            trace.add_edge(s.node.name, self.name)
        trace.register(self)
        for s in syms:
            trace.register(s.node)
        return Symbolic(self, self.output_schema, name=f"{self.name}/out")

    def _validate_symbolic(self, upstream: OutputSchema) -> None:
        self.input_schema.validate_against_output(upstream, node_name=self.name)

    def _eager_call(self, *args: Any) -> np.ndarray:
        X = self._coerce_inputs(args)
        return self.predict(X)

    def _coerce_inputs(self, args: Tuple) -> np.ndarray:
        if len(args) == 0:
            raise DTError(f"Node '{self.name}' called with no inputs.")
        if len(args) == 1:
            return to_numpy(args[0])
        # Multiple real inputs: column-concatenate (matches ConcatMerge).
        mats = [as_column(to_numpy(a)) for a in args]
        check_row_count(*mats, name=f"node '{self.name}' multi-input")
        return np.concatenate(mats, axis=1)

    # -- fit / predict --------------------------------------------------
    def fit(self, X: Any, y: Any) -> "BaseNode":
        if self.adapter is None:
            raise DTError(f"Node '{self.name}' has no trainable adapter (kind={self.kind}).")
        Xn = ensure_2d(to_numpy(X), name=f"node '{self.name}' fit input")
        self._validate_real_input(Xn)
        self.adapter.fit(Xn, y)
        self._check_output_dim_after_fit()
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self.adapter is None:
            raise DTError(f"Node '{self.name}' has no adapter for predict.")
        self.adapter.check_fitted(self.name)
        Xn = ensure_2d(to_numpy(X), name=f"node '{self.name}' predict input")
        self._validate_real_input(Xn)
        out = np.asanyarray(self.adapter.predict(Xn))
        return self._validate_real_output(out)

    def _validate_real_input(self, Xn: np.ndarray) -> None:
        exp = self.input_schema.n_features
        if exp is not None and Xn.shape[1] != exp:
            raise ShapeMismatchError(
                f"Node '{self.name}' expects {exp} features, but got {Xn.shape[1]} "
                f"(input shape {Xn.shape}). Check the upstream node's output kind "
                f"(prediction=1 col, proba=k cols) or fix the data."
            )

    def _validate_real_output(self, out: np.ndarray) -> np.ndarray:
        out = np.asanyarray(out)
        want = self.output_schema
        if want.kind == OutputKind.PREDICTION:
            if out.ndim == 2 and out.shape[1] == 1:
                out = out.reshape(-1)
            if out.ndim != 1:
                raise ShapeMismatchError(
                    f"Node '{self.name}' declares PREDICTION output (n,) but produced "
                    f"shape {out.shape}. Use output='proba'/'transform' for matrix outputs."
                )
        else:
            out = as_column(out, name=f"node '{self.name}' output")
            if want.dim is not None and out.shape[1] != want.dim:
                raise ShapeMismatchError(
                    f"Node '{self.name}' declares {want.kind.value} dim={want.dim} but "
                    f"produced shape {out.shape}."
                )
        return out

    def _check_output_dim_after_fit(self) -> None:
        dim = getattr(self.adapter, "output_dim_", None)
        want = self.output_schema
        if dim is None or want.kind == OutputKind.PREDICTION:
            return
        if want.dim is not None and int(dim) != int(want.dim):
            raise ShapeMismatchError(
                f"Node '{self.name}' declares {want.kind.value} dim={want.dim} but the "
                f"fitted estimator produces dim={dim}. Fix the declared dim "
                f"(e.g. n_classes for proba) or the estimator config."
            )
        # Fill in unknown dims so downstream symbolic validation stays exact.
        object.__setattr__  # (frozen-schema safe: we replace, not mutate)
        self.output_schema = OutputSchema(kind=want.kind, dim=int(dim), task=want.task, dtype=want.dtype)

    @property
    def is_fitted(self) -> bool:
        return bool(self.adapter is not None and self.adapter.is_fitted)

    def describe(self) -> str:
        return (
            f"{self.name} [{self.kind}]: "
            f"in(n_features={self.input_schema.n_features}) -> "
            f"out({self.output_schema.describe()})"
        )

    # -- serialization helpers ------------------------------------------
    def meta(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "input_schema": {
                "n_features": self.input_schema.n_features,
                "dtype": self.input_schema.dtype,
                "task": self.input_schema.task.value,
            },
            "output_schema": {
                "kind": self.output_schema.kind.value,
                "dim": self.output_schema.dim,
                "task": self.output_schema.task.value,
                "dtype": self.output_schema.dtype,
            },
            "adapter": self.adapter.config() if self.adapter else None,
        }


class InputNode(BaseNode):
    kind = "input"

    def __init__(self, *, name: str = "input", n_features: Optional[int] = None, dtype: Optional[str] = None):
        super().__init__(
            name=name,
            input_schema=FeatureSchema(n_features=None),
            output_schema=OutputSchema(kind=OutputKind.FEATURES, dim=n_features, task=TaskType.FEATURES, dtype=dtype),
        )
        self.n_features = n_features

    def _trace_call(self, *args: Any) -> Symbolic:  # pragma: no cover - InputNode is placed, not called
        raise DTError("InputNode cannot be called; it is the graph entry point.")

    def produce_symbolic(self, trace: Any) -> Symbolic:
        trace.register(self)
        return Symbolic(self, self.output_schema, name=f"{self.name}/out")

    def fit(self, X: Any, y: Any = None) -> "InputNode":  # type: ignore[override]
        Xn = ensure_2d(to_numpy(X), name=f"input '{self.name}'")
        if self.n_features is not None and Xn.shape[1] != self.n_features:
            raise ShapeMismatchError(
                f"Input '{self.name}' expects {self.n_features} features, but got "
                f"{Xn.shape[1]} (shape {Xn.shape})."
            )
        if self.output_schema.dim is None:
            self.output_schema = OutputSchema(
                kind=OutputKind.FEATURES, dim=int(Xn.shape[1]), task=TaskType.FEATURES, dtype=self.output_schema.dtype
            )
        return self

    def predict(self, X: Any) -> np.ndarray:  # type: ignore[override]
        Xn = ensure_2d(to_numpy(X), name=f"input '{self.name}'")
        if self.n_features is not None and Xn.shape[1] != self.n_features:
            raise ShapeMismatchError(
                f"Input '{self.name}' expects {self.n_features} features, but got "
                f"{Xn.shape[1]} (shape {Xn.shape})."
            )
        return Xn


class _EstimatorNode(BaseNode):
    """Shared constructor for estimator-backed nodes."""

    def __init__(
        self,
        adapter: BaseAdapter,
        *,
        name: Optional[str] = None,
        in_features: Optional[int] = None,
        out_dim: Optional[int] = None,
        output: str = "prediction",
        task: TaskType = TaskType.UNSPECIFIED,
    ):
        kind_map = {
            "prediction": OutputKind.PREDICTION,
            "proba": OutputKind.PROBA,
            "transform": OutputKind.TRANSFORM,
            "logits": OutputKind.LOGITS,
        }
        if output not in kind_map:
            raise DTError(f"Unknown output='{output}' for node '{name}'. Use prediction|proba|transform|logits.")
        okind = kind_map[output]
        dim = None if okind == OutputKind.PREDICTION else out_dim
        super().__init__(
            name=name,
            input_schema=FeatureSchema(n_features=in_features),
            output_schema=OutputSchema(kind=okind, dim=dim, task=task),
        )
        self.adapter = adapter


class SklearnNode(_EstimatorNode):
    kind = "sklearn"

    def __init__(self, estimator: Any, *, name=None, in_features=None, out_dim=None,
                 output: str = "prediction", task: TaskType = TaskType.UNSPECIFIED):
        super().__init__(SklearnAdapter(estimator, output=output), name=name or _auto_name("sklearn_"),
                         in_features=in_features, out_dim=out_dim, output=output, task=task)


class CatBoostNode(_EstimatorNode):
    kind = "catboost"

    def __init__(self, estimator: Any, *, name=None, in_features=None, out_dim=None,
                 output: str = "prediction", task: TaskType = TaskType.UNSPECIFIED):
        super().__init__(CatBoostAdapter(estimator, output=output), name=name or _auto_name("cb_"),
                         in_features=in_features, out_dim=out_dim, output=output, task=task)


class XGBoostNode(_EstimatorNode):
    kind = "xgboost"

    def __init__(self, estimator: Any, *, name=None, in_features=None, out_dim=None,
                 output: str = "prediction", task: TaskType = TaskType.UNSPECIFIED):
        super().__init__(XGBoostAdapter(estimator, output=output), name=name or _auto_name("xgb_"),
                         in_features=in_features, out_dim=out_dim, output=output, task=task)


class TorchNode(_EstimatorNode):
    kind = "torch"

    def __init__(self, module: Any, *, name=None, in_features=None, out_dim=None,
                 output: str = "prediction", task: TaskType = TaskType.UNSPECIFIED,
                 loss_fn=None, optimizer_fn=None, epochs=10, batch_size=32, lr=1e-3, device=None):
        # Torch modules naturally emit matrices; a PREDICTION-declared torch
        # node must emit (n,1) or (n,) -- validated at runtime.
        super().__init__(
            TorchAdapter(module, loss_fn=loss_fn, optimizer_fn=optimizer_fn, epochs=epochs,
                         batch_size=batch_size, lr=lr, device=device),
            name=name or _auto_name("torch_"),
            in_features=in_features, out_dim=out_dim, output=output, task=task,
        )


class MergeNode(BaseNode):
    kind = "merge"
    op: str = "concat"

    def __init__(self, *, name: Optional[str] = None, input_schemas: Optional[list] = None):
        # Merge accepts ANY widths (validated at runtime: same row count).
        super().__init__(
            name=name,
            input_schema=FeatureSchema(n_features=None),
            output_schema=OutputSchema(kind=OutputKind.FEATURES, dim=None, task=TaskType.FEATURES),
        )

    def _validate_symbolic(self, upstream: OutputSchema) -> None:
        return  # any upstream kind accepted; widths checked after fit / at runtime

    def _validate_real_input(self, Xn: np.ndarray) -> None:
        return


class ConcatMerge(MergeNode):
    """Column-concatenate branch outputs: (n,a)+(n,b) -> (n,a+b)."""

    kind = "concat"
    op = "concat"

    def __init__(self, *, name: Optional[str] = None):
        super().__init__(name=name or _auto_name("concat_"))

    def _trace_call(self, *args: Any) -> Symbolic:
        syms = _as_symbolic_list(args)
        trace = _current_trace.get()
        if trace is None:
            raise DTError("ConcatMerge Symbolic call outside graph trace.")
        total: Optional[int] = None
        widths = [s.schema.width() for s in syms]
        if all(w is not None for w in widths):
            total = int(sum(widths))  # type: ignore[arg-type]
        self.output_schema = OutputSchema(kind=OutputKind.FEATURES, dim=total, task=TaskType.FEATURES)
        for s in syms:
            trace.add_edge(s.node.name, self.name)
        trace.register(self)
        for s in syms:
            trace.register(s.node)
        return Symbolic(self, self.output_schema)

    def _eager_call(self, *args: Any) -> np.ndarray:
        mats = [as_column(to_numpy(a)) for a in args]
        check_row_count(*mats, name=f"merge '{self.name}'")
        return np.concatenate(mats, axis=1)

    def fit(self, X: Any, y: Any = None) -> "ConcatMerge":  # type: ignore[override]
        return self

    def predict(self, X: Any = None) -> np.ndarray:  # type: ignore[override]
        raise DTError("ConcatMerge.predict needs branch values; it runs inside graph execution.")


class _ReduceMerge(MergeNode):
    op: str = "mean"

    def _trace_call(self, *args: Any) -> Symbolic:
        syms = _as_symbolic_list(args)
        trace = _current_trace.get()
        if trace is None:
            raise DTError("Merge Symbolic call outside graph trace.")
        widths = {s.schema.width() for s in syms}
        if len(widths) != 1 or None in widths:
            raise ShapeMismatchError(
                f"Merge '{self.name}' (op={self.op}) needs all branches to have the "
                f"same known width; got {[s.schema.describe() for s in syms]}."
            )
        w = widths.pop()
        self.output_schema = OutputSchema(kind=OutputKind.FEATURES, dim=w, task=TaskType.FEATURES)
        for s in syms:
            trace.add_edge(s.node.name, self.name)
        trace.register(self)
        for s in syms:
            trace.register(s.node)
        return Symbolic(self, self.output_schema)

    def _eager_call(self, *args: Any) -> np.ndarray:
        mats = [np.asanyarray(to_numpy(a), dtype=float) for a in args]
        check_row_count(*mats, name=f"merge '{self.name}'")
        cols = {m.shape[1] if m.ndim == 2 else 1 for m in mats}
        if len(cols) != 1:
            raise ShapeMismatchError(f"Merge '{self.name}' needs equal widths, got {[m.shape for m in mats]}.")
        stack = np.stack([m.reshape(m.shape[0], -1) for m in mats], axis=0)
        if self.op == "sum":
            return stack.sum(axis=0)
        return stack.mean(axis=0)

    def fit(self, X: Any, y: Any = None) -> "_ReduceMerge":  # type: ignore[override]
        return self

    def predict(self, X: Any = None) -> np.ndarray:  # type: ignore[override]
        raise DTError("ReduceMerge runs inside graph execution.")


class SumMerge(_ReduceMerge):
    kind = "sum"
    op = "sum"

    def __init__(self, *, name: Optional[str] = None):
        super().__init__(name=name or _auto_name("sum_"))


class MeanMerge(_ReduceMerge):
    kind = "mean"
    op = "mean"

    def __init__(self, *, name: Optional[str] = None):
        super().__init__(name=name or _auto_name("mean_"))


class FunctionNode(BaseNode):
    """Custom model / lambda node: ``fn(*branch_arrays) -> array``.

    Use for custom models, projections (e.g. slicing proba columns), or glue
    code. ``in_features``/``out_dim``/``output`` declare the contract exactly
    like estimator nodes.
    """

    kind = "function"

    def __init__(self, fn: Callable[..., Any], *, name=None, in_features=None,
                 out_dim=None, output: str = "prediction", task: TaskType = TaskType.UNSPECIFIED):
        kind_map = {
            "prediction": OutputKind.PREDICTION,
            "proba": OutputKind.PROBA,
            "transform": OutputKind.TRANSFORM,
            "features": OutputKind.FEATURES,
            "logits": OutputKind.LOGITS,
        }
        if output not in kind_map:
            raise DTError(f"Unknown output='{output}' for FunctionNode.")
        okind = kind_map[output]
        super().__init__(
            name=name or _auto_name("fn_"),
            input_schema=FeatureSchema(n_features=in_features),
            output_schema=OutputSchema(kind=okind, dim=None if okind == OutputKind.PREDICTION else out_dim, task=task),
        )
        self.fn = fn
        self._fitted_fn = False

    def fit(self, X: Any, y: Any = None) -> "FunctionNode":  # type: ignore[override]
        Xn = ensure_2d(to_numpy(X), name=f"function node '{self.name}' fit input")
        self._validate_real_input(Xn)
        if hasattr(self.fn, "fit"):  # custom estimator with fit/predict
            self.fn.fit(Xn, None if y is None else np.asanyarray(y))
        self._fitted_fn = True
        # Probe output dim for matrix kinds.
        if self.output_schema.kind != OutputKind.PREDICTION:
            try:
                out = np.asanyarray(self.fn(Xn[: min(5, len(Xn))]))
                out = as_column(out)
                self.output_schema = OutputSchema(kind=self.output_schema.kind, dim=int(out.shape[1]),
                                                  task=self.output_schema.task)
            except Exception:
                pass
        return self

    def predict(self, X: Any) -> np.ndarray:  # type: ignore[override]
        if hasattr(self.fn, "predict") and not callable(getattr(self.fn, "__call__", None)):
            pass
        Xn = ensure_2d(to_numpy(X), name=f"function node '{self.name}' predict input")
        self._validate_real_input(Xn)
        if hasattr(self.fn, "predict") and not isinstance(self.fn, type) and hasattr(self.fn, "fit"):
            out = np.asanyarray(self.fn.predict(Xn))
        else:
            out = np.asanyarray(self.fn(Xn))
        return self._validate_real_output(out)

    @property
    def is_fitted(self) -> bool:  # type: ignore[override]
        if hasattr(self.fn, "fit"):
            return self._fitted_fn
        return True  # pure function: always usable

    def meta(self) -> dict:
        d = super().meta()
        d["adapter"] = {"kind": "function", "fn": getattr(self.fn, "__name__", type(self.fn).__name__)}
        return d
