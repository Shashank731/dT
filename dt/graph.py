"""Graph / dT: the typed DAG abstraction + executor.

Design summary (see README for full reasoning):

- The correct abstraction is a DAG (Pipeline = linear special case, Tree
  forbids diamond merges, "computational graph" overpromises autograd).
- ``dT`` is NOT an ``nn.Module``. It is a standalone graph container that
  *holds* TorchNodes. Inheriting nn.Module would lie about parameters(),
  backward(), and state_dict for tree models.
- Users write PyTorch-familiar code::

      class MyModel(dT):
          def __init__(self):
              super().__init__()
              self.cb = CatBoostNode(...)
              self.xgb = XGBoostNode(...)
          def forward(self, x):
              return self.xgb(self.cb(x))

  ``forward`` is traced once with Symbolic placeholders to capture the DAG
  (nodes + edges + schemas), then the executor runs the DAG topologically
  with real data for fit/predict. Branching/merging is plain Python code
  inside forward plus Merge nodes.
- Training = sequential topological fit. Gradients never cross tree
  boundaries (honest limitation, enforced by architecture).
"""

from __future__ import annotations

import inspect
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .data import as_column, ensure_2d, target_to_numpy, to_numpy
from .exceptions import (
    CycleError,
    DTError,
    DataError,
    SchemaMismatchError,
    SerializationError,
    ShapeMismatchError,
    UnfittedNodeError,
)
from .node import (
    BaseNode,
    ConcatMerge,
    InputNode,
    Symbolic,
    _current_trace,
)
from .schema import OutputKind


class _Trace:
    """Collects nodes/edges while forward() runs on Symbolics."""

    def __init__(self):
        self.nodes: Dict[str, BaseNode] = {}
        self.edges: List[Tuple[str, str]] = []  # (producer, consumer)
        self.inputs: Dict[str, InputNode] = {}
        self.output_symbolic: Any = None

    def register(self, node: BaseNode) -> None:
        self.nodes[node.name] = node
        if isinstance(node, InputNode):
            self.inputs[node.name] = node

    def add_edge(self, producer: str, consumer: str) -> None:
        if (producer, consumer) not in self.edges:
            self.edges.append((producer, consumer))

    def topo_order(self) -> List[str]:
        indeg: Dict[str, int] = {n: 0 for n in self.nodes}
        children: Dict[str, List[str]] = defaultdict(list)
        for p, c in self.edges:
            if p not in indeg:
                raise DTError(f"Edge from unknown node '{p}'.")
            if c not in indeg:
                raise DTError(f"Edge to unknown node '{c}'.")
            children[p].append(c)
            indeg[c] += 1
        q = deque([n for n, d in indeg.items() if d == 0])
        order: List[str] = []
        while q:
            n = q.popleft()
            order.append(n)
            for c in children[n]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    q.append(c)
        if len(order) != len(self.nodes):
            cyclic = [n for n, d in indeg.items() if d > 0]
            raise CycleError(
                f"Graph contains a cycle involving {cyclic}. dT graphs must be "
                f"acyclic (DAG): check forward() for feedback loops."
            )
        return order

    def parents_of(self, name: str) -> List[str]:
        return [p for p, c in self.edges if c == name]


class Graph:
    """Explicit DAG container. Usually subclassed as ``dT`` with forward()."""

    def __init__(self):
        self._nodes: Dict[str, BaseNode] = {}
        self._trace_cache: Optional[_Trace] = None
        self._input_names: List[str] = []

    # -- node registration (attribute assignment hooks nodes) ------------
    def __setattr__(self, key: str, value: Any) -> None:
        if isinstance(value, BaseNode) and not key.startswith("_"):
            nodes = self.__dict__.get("_nodes")
            if nodes is not None:
                # Rename auto-named nodes to the attribute name for clarity.
                if key not in nodes:
                    value.name = key
                nodes[key] = value
        super().__setattr__(key, value)

    def add_node(self, node: BaseNode, *, name: Optional[str] = None) -> BaseNode:
        """Explicit (non-forward) node registration for the low-level API."""
        n = name or node.name
        node.name = n
        self._nodes[n] = node
        try:
            super().__setattr__(n, node)
        except Exception:
            pass
        self._trace_cache = None
        return node

    @property
    def nodes(self) -> Dict[str, BaseNode]:
        return dict(self._nodes)

    # -- tracing --------------------------------------------------------
    def forward(self, *args: Any) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError("Define forward() in your dT subclass.")

    def _forward_params(self) -> List[str]:
        try:
            sig = inspect.signature(self.forward)
        except (TypeError, ValueError):
            return ["x"]
        return [p for p in sig.parameters if p != "self"] or ["x"]

    def _trace_(self) -> _Trace:
        if self._trace_cache is not None:
            return self._trace_cache
        trace = _Trace()
        for node in self._nodes.values():
            trace.register(node)
        params = self._forward_params()
        token = _current_trace.set(trace)
        try:
            syms: Dict[str, Symbolic] = {}
            for p in params:
                inp = trace.inputs.get(p)
                if inp is None:
                    inp = InputNode(name=p)
                    trace.register(inp)
                    # keep graph-level reference so serialization sees it
                    self._nodes.setdefault(p, inp)
                syms[p] = inp.produce_symbolic(trace)
            out = self.forward(*[syms[p] for p in params])
            trace.output_symbolic = out
            self._validate_trace_outputs(trace, out)
        finally:
            _current_trace.reset(token)
        self._trace_cache = trace
        self._input_names = params
        return trace

    @staticmethod
    def _validate_trace_outputs(trace: _Trace, out: Any) -> None:
        def check(s: Any) -> None:
            if isinstance(s, Symbolic):
                if s.node.name not in trace.nodes:
                    raise DTError(f"forward() returned Symbolic from untracked node '{s.node.name}'.")
            elif isinstance(s, (list, tuple, dict)):
                vals = s if not isinstance(s, dict) else s.values()
                for v in vals:
                    check(v)
            elif s is None:
                raise DTError("forward() returned None. Return a Node output (Symbolic).")
            else:
                raise DTError(
                    f"forward() must return Node outputs (Symbolic), got {type(out).__name__}. "
                    f"Did you forget to call a node, e.g. 'return self.xgb(h)'?"
                )

        check(out)

    # -- data binding ----------------------------------------------------
    def _bind_inputs(self, X: Any, params: List[str]) -> Dict[str, np.ndarray]:
        if len(params) == 1:
            if isinstance(X, dict) and params[0] in X:
                raw = X[params[0]]
            else:
                raw = X
            return {params[0]: ensure_2d(to_numpy(raw), name=f"graph input '{params[0]}'")}
        if not isinstance(X, dict):
            raise DataError(
                f"Graph takes {len(params)} inputs {params}; pass a dict "
                f"{{name: array}}, e.g. graph.predict({{{params[0]}: X1, ...}})."
            )
        missing = [p for p in params if p not in X]
        if missing:
            raise DataError(f"Missing graph inputs {missing}; got keys {sorted(X.keys())}.")
        return {p: ensure_2d(to_numpy(X[p]), name=f"graph input '{p}'") for p in params}

    def _bind_targets(self, y: Any, params: List[str]) -> Dict[str, np.ndarray]:
        """MVP: one global y shared by every trainable node (sequential fit).

        Per-node targets (dict) are accepted and validated but default to the
        global target -- see README 'Training semantics'.
        """
        if isinstance(y, dict):
            return {k: target_to_numpy(v) for k, v in y.items()}
        arr = target_to_numpy(y)
        return {"__global__": arr}

    def _target_for(self, node_name: str, targets: Dict[str, np.ndarray]) -> np.ndarray:
        if node_name in targets:
            return targets[node_name]
        if "__global__" in targets:
            return targets["__global__"]
        raise DTError(
            f"No target for node '{node_name}'. Pass a global y or a dict with "
            f"key '{node_name}'."
        )

    def _features_of(self, value: np.ndarray, *, node_name: str) -> np.ndarray:
        """Coerce a stored node output to a 2D feature matrix for downstream
        use. PREDICTION outputs are (n,) and become (n, 1)."""
        arr = np.asanyarray(value)
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        if arr.ndim == 2:
            return arr
        raise DataError(
            f"Node output feeding '{node_name}' has shape {arr.shape}; "
            f"expected (n,) or (n, k)."
        )

    # -- execution --------------------------------------------------------
    def _materialize(self, trace: _Trace, order: List[str], values: Dict[str, np.ndarray],
                     *, trainable_only: bool = False) -> Dict[str, np.ndarray]:
        """Run nodes in topo order; each node executes exactly once (memoized
        in ``values``, which is pre-filled with input arrays)."""
        from .node import MergeNode

        for name in order:
            if name in values:
                continue
            node = trace.nodes[name]
            if isinstance(node, InputNode):
                raise DTError(f"Input '{name}' has no data bound. Pass X (or dict) to fit/predict.")
            if isinstance(node, MergeNode):
                parents = trace.parents_of(name)
                branch_vals = [np.asanyarray(values[p]) for p in parents]
                values[name] = node._eager_call(*branch_vals)
                continue
            parents = trace.parents_of(name)
            if len(parents) == 0:
                raise DTError(
                    f"Node '{name}' has no upstream connection in the traced graph. "
                    f"Every non-input node must be called on something inside forward()."
                )
            if len(parents) == 1:
                Xin = self._features_of(values[parents[0]], node_name=name)
            else:  # multi-parent without explicit Merge: concat (documented)
                mats = [as_column(np.asanyarray(values[p])) for p in parents]
                from .data import check_row_count

                check_row_count(*mats, name=f"node '{name}' multi-parent")
                Xin = np.concatenate(mats, axis=1)
            values[name] = node.predict(Xin)
        return values

    # -- public API --------------------------------------------------------
    def fit(self, X: Any, y: Any) -> "Graph":
        """Sequential topological fit.

        For nodes in topological order: materialize the node's input from
        already-fitted upstream nodes, then fit the node on ``(input, y)``.
        Upstream nodes are fitted before their inputs are used, so every
        node's training data is real (never symbolic).
        """
        trace = self._trace_()
        order = trace.topo_order()
        params = self._input_names or self._forward_params()
        inputs = self._bind_inputs(X, params)
        targets = self._bind_targets(y, params)
        n_rows = {k: v.shape[0] for k, v in inputs.items()}
        for k, t in targets.items():
            if t.shape[0] not in n_rows.values():
                raise DataError(
                    f"Target '{k}' has {t.shape[0]} rows but inputs have {n_rows}. "
                    f"X and y must have the same number of rows."
                )
        values: Dict[str, np.ndarray] = dict(inputs)
        from .node import FunctionNode, InputNode, MergeNode

        for name in order:
            node = trace.nodes[name]
            if isinstance(node, InputNode):
                node.fit(values[name])
                continue
            if isinstance(node, MergeNode):
                parents = trace.parents_of(name)
                branch_vals = [np.asanyarray(values[p]) for p in parents]
                values[name] = node._eager_call(*branch_vals)
                continue
            parents = trace.parents_of(name)
            if len(parents) == 1:
                Xin = self._features_of(values[parents[0]], node_name=name)
            else:
                mats = [as_column(np.asanyarray(values[p])) for p in parents]
                from .data import check_row_count

                check_row_count(*mats, name=f"node '{name}' multi-parent")
                Xin = np.concatenate(mats, axis=1)
            if isinstance(node, FunctionNode) and not hasattr(node.fn, "fit"):
                values[name] = node.predict(Xin)  # stateless glue: no fitting
                continue
            node.fit(Xin, self._target_for(name, targets))
            values[name] = node.predict(Xin)
        return self

    def predict(self, X: Any) -> Any:
        trace = self._trace_()
        order = trace.topo_order()
        params = self._input_names or self._forward_params()
        inputs = self._bind_inputs(X, params)
        # Fit InputNodes lazily so dims are known even if fit() was skipped
        # for frozen graphs (predict still requires fitted estimators).
        for pname, arr in inputs.items():
            node = trace.nodes.get(pname)
            if isinstance(node, InputNode) and node.output_schema.dim is None:
                node.fit(arr)
        values: Dict[str, np.ndarray] = dict(inputs)
        unfitted = [n for n, nd in trace.nodes.items()
                    if getattr(nd, "adapter", None) is not None and not nd.is_fitted
                    and not n in values]
        # FunctionNodes without fit are always usable; check the rest.
        real_unfitted = []
        for n in unfitted:
            from .node import FunctionNode

            nd = trace.nodes[n]
            if isinstance(nd, FunctionNode) and not hasattr(nd.fn, "fit"):
                continue
            real_unfitted.append(n)
        if real_unfitted:
            raise UnfittedNodeError(
                f"Cannot predict: nodes {real_unfitted} are not fitted. "
                f"Call graph.fit(X, y) first."
            )
        values = self._materialize(trace, order, values)
        return self._collect_output(trace.output_symbolic, values)

    def _collect_output(self, out: Any, values: Dict[str, np.ndarray]) -> Any:
        if isinstance(out, Symbolic):
            return values[out.node.name]
        if isinstance(out, list):
            return [self._collect_output(v, values) for v in out]
        if isinstance(out, tuple):
            return tuple(self._collect_output(v, values) for v in out)
        if isinstance(out, dict):
            return {k: self._collect_output(v, values) for k, v in out.items()}
        raise DTError(f"Unexpected traced output {type(out).__name__}.")

    def __call__(self, X: Any) -> Any:
        """``graph(X)`` = inference (predict). Training is always .fit()."""
        return self.predict(X)

    def describe(self) -> str:
        trace = self._trace_()
        lines = ["dT graph:", f"  inputs: {self._input_names}", "  nodes:"]
        for n in trace.topo_order():
            lines.append(f"    - {trace.nodes[n].describe()}")
        lines.append(f"  edges: {trace.edges}")
        return "\n".join(lines)

    # -- serialization ------------------------------------------------------
    def save(self, path: str) -> str:
        from .serialize import save_graph

        return save_graph(self, path)

    @classmethod
    def load(cls, path: str, model_cls: Optional[type] = None) -> "Graph":
        from .serialize import load_graph

        return load_graph(path, model_cls=model_cls or cls)


class dT(Graph):
    """Base class for user models. Subclass, assign Nodes in __init__, define
    forward(). See module docstring + README for the full contract."""

    pass
