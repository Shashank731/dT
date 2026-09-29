"""Serialization: zip archive (*.dt) with graph.json + per-node payloads.

Why not torch.save() alone: a graph contains CatBoost/XGBoost/sklearn
objects (each with its own native format), plus graph structure, schemas,
and merge/function wiring. One torch file cannot hold all of that.

Layout inside ``model.dt`` (a zip)::

    graph.json                  # class qualname, input names, nodes meta, edges
    nodes/<name>/...            # per-adapter payloads (cbm, xgb, pt, pkl)

Load: rebuilds the node objects from graph.json (adapters restored via
their native load), re-attaches them to an instance of the user class
(created via __new__ so __init__ estimators are not needed), and restores
the traced DAG. The user class must be importable for full fidelity;
otherwise a generic Graph is returned.
"""

from __future__ import annotations

import importlib
import io
import json
import os
import tempfile
import zipfile
from typing import Optional

from .exceptions import SerializationError
from .schema import FeatureSchema, OutputKind, OutputSchema, TaskType


def _qualname(obj: object) -> str:
    return type(obj).__module__ + "." + type(obj).__qualname__


def _import_qualname(qn: str):
    mod_name, _, attr = qn.rpartition(".")
    mod = importlib.import_module(mod_name)
    obj = mod
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def save_graph(graph: object, path: str) -> str:
    trace = graph._trace_()  # noqa: SLF001
    graph_json = {
        "format": "dT/0.1",
        "class": _qualname(graph),
        "inputs": graph._input_names or [],  # noqa: SLF001
        "nodes": [node.meta() for node in trace.nodes.values()],
        "edges": [list(e) for e in trace.edges],
    }
    # Collect output routing: which node(s) form forward's return value.
    def sym_ref(s):
        from .node import Symbolic

        if isinstance(s, Symbolic):
            return {"node": s.node.name}
        if isinstance(s, list):
            return [{"node": v.node.name} for v in s]
        if isinstance(s, tuple):
            return {"tuple": [v.node.name for v in s]}
        if isinstance(s, dict):
            return {"dict": {k: v.node.name for k, v in s.items()}}
        raise SerializationError(f"Cannot serialize traced output of type {type(s).__name__}.")

    graph_json["output"] = sym_ref(trace.output_symbolic)

    with tempfile.TemporaryDirectory() as tmp:
        node_dir = os.path.join(tmp, "nodes")
        os.makedirs(node_dir, exist_ok=True)
        for name, node in trace.nodes.items():
            from .node import FunctionNode

            ndir = os.path.join(node_dir, name)
            os.makedirs(ndir, exist_ok=True)
            if isinstance(node, FunctionNode):
                with open(os.path.join(ndir, "fn.json"), "w") as f:
                    json.dump({"name": node.name, "kind": node.kind, "op": "function",
                               "fn": getattr(node.fn, "__name__", "fn")}, f)
                # Stateful custom estimators: pickle whole object as fallback.
                try:
                    import pickle

                    with open(os.path.join(ndir, "fn.pkl"), "wb") as f:
                        pickle.dump(node.fn, f)
                except Exception:
                    pass
                continue
            if node.adapter is not None:
                node.adapter.save(ndir)
            else:  # Input / Merge: marker only
                with open(os.path.join(ndir, "marker.json"), "w") as f:
                    json.dump({"kind": node.kind, "op": getattr(node, "op", "")}, f)
        with open(os.path.join(tmp, "graph.json"), "w") as f:
            json.dump(graph_json, f, indent=2)
        # Zip it.
        if not path.endswith(".dt"):
            path = path + ".dt"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(tmp):
                for fn in files:
                    full = os.path.join(root, fn)
                    z.write(full, os.path.relpath(full, tmp))
    return path


def load_graph(path: str, model_cls: Optional[type] = None) -> object:
    from . import node as node_mod
    from .adapter import CatBoostAdapter, SklearnAdapter, TorchAdapter, XGBoostAdapter
    from .graph import Graph

    if not os.path.exists(path):
        raise SerializationError(f"File not found: {path}")
    try:
        z = zipfile.ZipFile(path, "r")
    except Exception as e:
        raise SerializationError(f"{path} is not a valid .dt archive: {e}") from e
    with tempfile.TemporaryDirectory() as tmp:
        z.extractall(tmp)
        try:
            with open(os.path.join(tmp, "graph.json")) as f:
                gj = json.load(f)
        except Exception as e:
            raise SerializationError(f"Archive {path} has no valid graph.json: {e}") from e
        if gj.get("format", "").split("/")[0] != "dT":
            raise SerializationError(f"Archive {path} is not a dT graph (format={gj.get('format')}).")

        # Resolve the user class: explicit arg wins, else stored qualname if
        # importable, else generic Graph.
        cls = model_cls
        if cls is None or cls is Graph or getattr(cls, "__name__", "") in ("Graph", "dT"):
            try:
                maybe = _import_qualname(gj["class"])
                if isinstance(maybe, type):
                    cls = maybe
                else:
                    cls = Graph
            except Exception:
                cls = Graph
        try:
            graph = cls.__new__(cls)
            Graph.__init__(graph)
        except Exception as e:
            raise SerializationError(f"Could not instantiate {cls}: {e}") from e

        meta_by_name = {m["name"]: m for m in gj["nodes"]}
        rebuilt = {}
        for m in gj["nodes"]:
            name, kind = m["name"], m["kind"]
            isch, osch = m["input_schema"], m["output_schema"]
            ndir = os.path.join(tmp, "nodes", name)
            if kind == "input":
                nd = node_mod.InputNode(name=name, n_features=osch["dim"])
            elif kind in ("concat", "merge", "sum", "mean"):
                nd = {"concat": node_mod.ConcatMerge, "sum": node_mod.SumMerge,
                      "mean": node_mod.MeanMerge, "merge": node_mod.ConcatMerge}[kind](name=name)
                if osch["dim"] is not None:
                    from .schema import OutputKind as OK

                    nd.output_schema = OutputSchema(kind=OK.FEATURES, dim=osch["dim"], task=TaskType.FEATURES)
            elif kind == "function":
                import pickle

                try:
                    with open(os.path.join(ndir, "fn.pkl"), "rb") as f:
                        fn = pickle.load(f)
                except Exception:
                    fn = (lambda X: X)
                nd = node_mod.FunctionNode(fn, name=name,
                                           out_dim=osch["dim"],
                                           output=osch["kind"] if osch["kind"] in
                                           ("prediction", "proba", "transform", "features", "logits") else "features")
            else:
                acfg = m.get("adapter") or {}
                aclass = {"sklearn": SklearnAdapter, "catboost": CatBoostAdapter,
                          "xgboost": XGBoostAdapter, "torch": TorchAdapter}[kind]
                holder = aclass.__new__(aclass)
                holder._fitted = False
                holder.output_dim_ = None
                try:
                    holder.load(ndir)
                except Exception as e:
                    raise SerializationError(f"Failed to load node '{name}': {e}") from e
                adapter = holder
                # Rebuild node shell with stored schemas.
                nd = node_mod.BaseNode.__new__(node_mod.BaseNode)
                from .schema import OutputKind as OK

                nd.name = name
                nd.kind = kind
                nd.input_schema = FeatureSchema(n_features=isch["n_features"], dtype=isch["dtype"],
                                                task=TaskType(isch["task"]))
                nd.output_schema = OutputSchema(kind=OK(osch["kind"]), dim=osch["dim"],
                                                task=TaskType(osch["task"]), dtype=osch["dtype"])
                # Re-attach the right subclass behavior.
                subclass = {"sklearn": node_mod.SklearnNode, "catboost": node_mod.CatBoostNode,
                            "xgboost": node_mod.XGBoostNode, "torch": node_mod.TorchNode}[kind]
                nd.__class__ = subclass
                nd.adapter = adapter
                if kind == "torch":
                    for k in ("epochs", "batch_size", "lr"):
                        pass
            rebuilt[name] = nd
            graph._nodes[name] = nd  # noqa: SLF001
            try:
                object.__setattr__(graph, name, nd)
            except Exception:
                pass

        # Restore traced DAG cache so predict() works without re-tracing
        # (re-tracing also works since forward() calls the same node names...
        # actually forward() is not re-run; we rebuild the trace directly).
        from .graph import _Trace

        trace = _Trace()
        for nd in rebuilt.values():
            trace.register(nd)
        trace.edges = [tuple(e) for e in gj["edges"]]
        out = gj.get("output", {})

        def resolve(ref):
            from .node import Symbolic

            if isinstance(ref, dict) and "node" in ref:
                nd = rebuilt[ref["node"]]
                return Symbolic(nd, nd.output_schema)
            if isinstance(ref, list):
                return [resolve(r) for r in ref]
            if isinstance(ref, dict) and "tuple" in ref:
                return tuple(resolve({"node": n}) for n in ref["tuple"])
            if isinstance(ref, dict) and "dict" in ref:
                return {k: resolve({"node": n}) for k, n in ref["dict"].items()}
            raise SerializationError(f"Cannot resolve output ref {ref}.")

        trace.output_symbolic = resolve(out)
        trace.inputs = {n: nd for n, nd in rebuilt.items() if isinstance(nd, node_mod.InputNode)}
        graph._trace_cache = trace  # noqa: SLF001
        graph._input_names = gj.get("inputs", list(trace.inputs.keys()))  # noqa: SLF001
        return graph
