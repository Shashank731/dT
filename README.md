# dT — typed DAG for heterogeneous ML models

Compose **PyTorch, CatBoost, XGBoost, sklearn, and custom models** into one
typed, executable graph with a PyTorch-familiar API.

```python
class MyModel(dT):
    def __init__(self):
        super().__init__()
        self.cb = CatBoostNode(CatBoostRegressor(iterations=100), in_features=12)
        self.xgb = XGBoostNode(XGBRegressor(n_estimators=100), in_features=1)

    def forward(self, x):
        return self.xgb(self.cb(x))   # Input(12) -> CatBoost -> XGBoost -> Output
```

```python
model.fit(X, y)
preds = model.predict(X)   # or model(X)
model.save("model.dt")
model = dT.load("model.dt")
```

> **Honest headline:** dT does *sequential topological fitting* with *validated
> schemas*. It does **not** backpropagate through trees. The architecture is
> built around that limitation instead of hiding it.

---

## 1. What dT is

dT is a small framework for composing **heterogeneous** models — differentiable
(torch) and non-differentiable (trees, sklearn) — into a single **typed DAG**
(directed acyclic graph) with:

- `Node` — a typed computation unit (`TorchNode`, `CatBoostNode`,
  `XGBoostNode`, `SklearnNode`, `FunctionNode`, `MergeNode`).
- `Adapter` — framework-specific logic (fit/predict/save per library). All
  CatBoost/XGBoost/torch details live here, never in the executor.
- `Schema` — explicit input/output contracts (`FeatureSchema`, `OutputSchema`,
  `TaskType`, `OutputKind`). Connections are validated **before execution**.
- `Graph`/`dT` — the DAG container + executor (trace → topological sort →
  sequential fit / memoized predict).
- Native serialization to a `.dt` zip (graph structure + schemas + each
  framework's native model format).

## 2. Motivation

Real pipelines are heterogeneous: gradient boosting for tabular features,
a torch head for representation learning, sklearn glue in between. Today you
must hand-wire conversions, shapes, and fitting order, and a silent
`(n,) vs (n,1) vs (n,k)` mismatch only explodes at 2 AM.

dT gives you:

1. One `fit / predict` over the whole DAG, in the right order.
2. Schema validation at graph-build time with actionable errors
   (`expects 4 features, but upstream node produces 7`).
3. Branching/merging as plain Python in `forward()`.
4. One-file serialization across frameworks.

## 3. Architecture

### 3.1 DAG, not Pipeline/Tree/"computational graph"

| Abstraction | Verdict |
|---|---|
| `Pipeline` (sklearn) | Linear only — cannot express the diamond merge. Rejected as the core. |
| `Tree` | Forbids merging (a node with two parents). Rejected. |
| Differentiable "computational graph" (torch autograd) | Overpromises: gradients cannot flow through CatBoost/XGBoost. Rejected as terminology. |
| **DAG with a typed executor** | Branching ✓, merging ✓, multi-input/output ✓, no autograd claims. **Chosen.** |

A Pipeline is just a linear DAG here — supported for free.

### 3.2 `dT` is NOT an `nn.Module`

```python
# deliberately NOT this:
class dT(nn.Module): ...
```

Inheriting `nn.Module` would imply `parameters()`, `.to()/.cuda()`,
`backward()`, and torch `state_dict` semantics for CatBoost/XGBoost — all
lies. Instead **`dT` is a standalone graph container that *holds*
`TorchNode`s** (which wrap real `nn.Module`s). Torch interop happens inside
`TorchAdapter` (local training loop, device handling, tensor conversion), and
`graph.predict()` accepts/returns numpy, pandas, or torch transparently.

### 3.3 How `forward()` works (trace, then execute)

1. **Trace:** `fit/predict/save` runs your `forward()` once with `Symbolic`
   placeholders. Each `Node.__call__` validates schemas symbolically and
   records `(producer → consumer)` edges. The result is a DAG + topological
   order (cycles → `CycleError`).
2. **Fit:** nodes execute in topo order; each node's input is materialized
   from already-fitted upstream nodes, then the node fits on `(input, y)`.
3. **Predict:** the same order replays with memoized values — every node runs
   **exactly once** per call, shared branches included.

```
Input(12)
   ↓
CatBoost (prediction: (n,) → 1 feature)
   ↓
XGBoost (in=1)
   ↓
Output
```

Branching is just Python + a merge node (see §5).

### 3.4 Project structure

```
dT/
├── dt/
│   ├── __init__.py     # public API
│   ├── schema.py       # FeatureSchema / OutputSchema / TaskType / OutputKind
│   ├── exceptions.py   # SchemaMismatchError, UnfittedNodeError, ...
│   ├── data.py         # numpy <-> pandas <-> torch conversion layer
│   ├── adapter.py      # Torch/CatBoost/XGBoost/Sklearn adapters
│   ├── node.py         # Node types, Symbolic, Merge nodes, FunctionNode
│   ├── graph.py        # dT/Graph: tracing, topo executor, fit/predict
│   └── serialize.py    # .dt zip format (graph.json + native payloads)
├── tests/test_dt.py    # 20 tests
├── examples/           # sequential.py, dag.py, torch_chain.py
└── README.md
```

## 4. Sequential example

```python
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from dt import dT, CatBoostNode, XGBoostNode

class MyModel(dT):
    def __init__(self):
        super().__init__()
        self.cb = CatBoostNode(CatBoostRegressor(iterations=100), in_features=12)
        self.xgb = XGBoostNode(XGBRegressor(n_estimators=100), in_features=1)

    def forward(self, x):
        return self.xgb(self.cb(x))

model = MyModel()
model.fit(X, y)          # fits cb on (X, y), then xgb on (cb(X), y)
print(model.predict(X))  # or model(X)
print(model.describe())  # nodes, schemas, edges in topo order
```

`12 → 1 → 1` is explicit: CatBoost's `PREDICTION` output `(n,)` counts as
**one** downstream feature.

## 5. DAG example (branching + merging)

```python
from sklearn.linear_model import LinearRegression
from dt import dT, CatBoostNode, XGBoostNode, SklearnNode, ConcatMerge

class Diamond(dT):
    def __init__(self):
        super().__init__()
        self.cb = CatBoostNode(CatBoostRegressor(iterations=100), in_features=12)
        self.xgb = XGBoostNode(XGBRegressor(n_estimators=100), in_features=12)
        self.merge = ConcatMerge()          # (n,1)+(n,1) -> (n,2)
        self.head = SklearnNode(LinearRegression(), in_features=2)

    def forward(self, x):
        a = self.cb(x)
        b = self.xgb(x)
        return self.head(self.merge(a, b))
```

```
             Input
            /     \
      CatBoost   XGBoost
           \     /
         ConcatMerge
               |
        LinearRegression
               |
             Output
```

Merges: `ConcatMerge` (default), `SumMerge`, `MeanMerge` (require equal
widths). Multi-input (`forward(self, x1, x2)` + `dict` data) and multi-output
(return tuple/dict) are supported.

## 6. Nodes, adapters, schemas

### The "4 outputs" contract (read this!)

`CatBoost: 12 → 4` is **not** `nn.Linear(12, 4)`. For tree models the number
means one of:

| Declaration | Meaning | Shape |
|---|---|---|
| `output="prediction"` (default) | 1 estimator's labels/values | `(n,)` → width 1 |
| `output="proba", out_dim=k` | per-class probabilities | `(n, k)` |
| `output="transform", out_dim=k` | k estimators' predictions / leaf features as downstream features | `(n, k)` |

```python
CatBoostNode(est, in_features=12)                          # 12 -> 1
CatBoostNode(est, in_features=12, output="proba", out_dim=3)  # 12 -> 3
```

Schemas are checked **twice**: symbolically at trace time (declared dims) and
against real data at runtime (fitted estimator's actual dim is probed after
`fit` and must match, else `ShapeMismatchError`).

```text
ShapeMismatchError: Node 'xgb' expects 4 features, but upstream node produces 7
(kind=proba, dim=7). Fix the upstream output (e.g. use proba vs prediction,
or insert a projection/merge) or change this node's expected input width.
```

### Node catalog (v0.1)

| Node | Wraps | Status |
|---|---|---|
| `TorchNode` | `nn.Module` + local optimizer loop | ✅ implemented |
| `CatBoostNode` | `CatBoostRegressor/Classifier` | ✅ implemented |
| `XGBoostNode` | `XGBRegressor/Classifier` (sklearn API) | ✅ implemented |
| `SklearnNode` | any sklearn estimator (`prediction/proba/transform`) | ✅ implemented |
| `FunctionNode` | lambda or custom estimator (`fit/predict` or pure fn) | ✅ implemented |
| `ConcatMerge/SumMerge/MeanMerge` | branch merging | ✅ implemented |
| LightGBM node | — | 🔬 planned (same adapter pattern) |
| Raw `xgboost.Booster` | — | ❌ unsupported (use sklearn API) |

Framework code lives in `Adapter` subclasses; the graph executor never imports
catboost/xgboost/torch directly (except the data layer).

## 7. CatBoost / XGBoost usage

```python
CatBoostNode(CatBoostClassifier(iterations=200), in_features=12,
             output="proba", out_dim=3)   # 3-class proba -> (n,3)
XGBoostNode(XGBClassifier(n_estimators=200), in_features=12,
            output="proba", out_dim=2)    # binary proba -> (n,2)
```

Notes:

- Pass **constructed estimator instances**; dT does not re-wrap hyperparams.
- v0.1 converts pandas → numpy at node boundaries: **numeric data only**.
  Native categorical passthrough is roadmap (the adapter boundary is where it
  will slot in).
- `predict` before `fit` raises `UnfittedNodeError` telling you to call
  `graph.fit(X, y)`.

## 8. Training semantics

`model.fit(X, y)` = **sequential topological fit**:

1. Topo-sort the traced DAG.
2. For each node in order: materialize its input from fitted upstream outputs,
   fit it on `(input, y)` where `y` is the **global target** (or `y[node_name]`
   if you pass a dict of per-node targets).
3. `TorchNode`s run a local mini-batch loop (`epochs/batch_size/lr`, optional
   custom `loss_fn`/`optimizer_fn`) on their own inputs. **No gradient crosses
   any node boundary, ever.**

`predict / transform / forward` split:

- `forward()` — *your* code, defines the DAG (runs on Symbolics at trace,
  never called directly with data).
- `fit(X, y)` — trains every trainable node in order.
- `predict(X)` / `graph(X)` — inference; each node once, memoized.
- No `backward()` exists on graphs. `TorchNode` has gradients *internally*;
  the graph does not differentiate *through* anything.

### Leakage warning (important)

Naive sequential fitting trains downstream nodes on **in-sample** upstream
predictions — optimistic for stacking. v0.1 does this plainly and documents
it. Out-of-fold (cross-validated) stacking predictions are **planned, not
implemented** — for rigorous stacking today, generate OOF features yourself
and feed them as a second graph input.

## 9. Inference & data

- `graph.predict(X)` / `graph(X)`; multi-input graphs take `dict`s.
- Accepted: `numpy.ndarray`, `pandas.DataFrame/Series`, `torch.Tensor`,
  array-likes. Converted once per node boundary (`torch.from_numpy` shares
  memory where possible; no per-row copies).
- Rules: batch dim first, 2D inputs (`(n, d)`; `(n,)` targets OK), consistent
  row counts across branches (else `DataError`), CPU default with CUDA when
  available for torch nodes.
- Unfitted graphs raise `UnfittedNodeError`, not garbage predictions.

## 10. Serialization

```python
model.save("model.dt")        # zip: graph.json + native payloads
model2 = dT.load("model.dt")  # or Graph.load(path)
```

`torch.save()` alone is insufficient (it knows nothing of CatBoost/XGBoost
files, graph edges, or schemas), so `.dt` is a zip: `graph.json` (class,
nodes, schemas, edges, output routing) + `nodes/<name>/` payloads
(CatBoost `.cbm`, XGBoost `.ubj`, torch `state_dict`, sklearn pickle).
Reload re-imports your model class when importable (define it at module
level, as in `examples/`), else returns a generic `Graph`.

Limitation: torch modules reload via **zero-arg constructor + state_dict**.
Modules needing constructor args must be re-registered in code first.

## 11. Installation

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas scikit-learn catboost xgboost pytest
pip install -e .            # from the repo root
pytest                      # 20 tests, should all pass
```

## 12. Examples

- `examples/sequential.py` — `CatBoost → XGBoost` (`12 → 1 → 1`).
- `examples/dag.py` — diamond `CatBoost/XGBoost → Concat → LinearRegression`.
- `examples/torch_chain.py` — `MLP → LinearRegression` (incl. serde-safe
  module-level classes).

## 13. Roadmap

- [ ] LightGBM node/adapter (same pattern; needs optional dep handling)
- [ ] Out-of-fold stacking predictions (`fit_stacking` with KFold)
- [ ] Categorical passthrough (pandas → CatBoost pools, no numpy coercion)
- [ ] Leaf-index / embedding `TRANSFORM` outputs for trees
- [ ] Per-node loss/target routing DSL, early stopping, eval sets
- [ ] ONNX export of the inference DAG (torch subgraphs + sklearn trees)
- [ ] GPU/Dask execution backends

## 14. Non-goals

- End-to-end backprop through trees (mathematically impossible for
  split-based GBDTs; we won't pretend).
- Distributed training, serving infra, HPO (use Optuna/Ray on top).
- AutoML / automatic graph search.
- Being an `nn.Module` or an sklearn `Pipeline` clone.

## 15. Comparison with existing solutions

| Tool | What it does | What dT adds |
|---|---|---|
| sklearn `Pipeline` | linear `fit/transform/predict` chain | arbitrary DAGs, torch/tree natives, cross-framework serde |
| `StackingClassifier/Regressor`, `Voting` | fixed 2-level ensembling | N-level DAGs, merges, multi-input/output, schemas |
| `ColumnTransformer` / `FeatureUnion` | parallel column transforms | full model DAG (estimators as features *and* heads) |
| `nn.Module` / `nn.Sequential` / `torch.fx` | differentiable graphs | heterogeneous graphs incl. non-differentiable models |
| ONNX | inference interchange | training semantics (`fit`) + native-orientation authoring |
| Keras Functional API | static DAG of differentiable layers | same *shape* of API, but heterogeneous + schema-typed |
| Dask / Prefect / Airflow | generic task DAGs | ML contracts: schemas, fit/predict split, OOF-leakage model |

**Is it redundant?** No — nothing above gives you *one typed, trainable,
serializable DAG across torch + GBDTs + sklearn*. The honest scope is narrow
(typed composition + sequential fitting, not autograd magic), which is exactly
why it earns its place.

## 16. Limitations (v0.1, plainly)

1. No gradients through trees; `TorchNode`s train locally only.
2. Naive in-sample stacking (no OOF yet) — see §8 leakage warning.
3. Numeric data only (pandas converted to numpy at boundaries).
4. Single global `y` unless you pass per-node target dicts.
5. Torch reload needs zero-arg module constructors.
6. No LightGBM, no raw Booster, no GPU placement DSL, no OOF/early-stopping plumbing.

## 17. Final verdict: **BUILD A SMALL MVP** ✅

The concept is sound but narrow: composition + contracts + sequential fitting
is genuinely useful and unserved; end-to-end differentiation across GBDTs is
impossible and must stay out of scope. This repo is that MVP: DAG execution,
schemas with clear errors, four node families, tested serialization — and an
explicit non-goals list to keep it honest. Next highest-value step: OOF
stacking, then LightGBM + categorical passthrough.

**Feature status legend:** ✅ implemented · 🔬 experimental · 📋 planned ·
❌ unsupported (see tables above).
