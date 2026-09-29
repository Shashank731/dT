"""Data-contract layer: numpy <-> pandas <-> torch conversions.

Canonical rules (v0.1):
- Tree/sklearn adapters operate on ``np.ndarray`` (2D, numeric). Pandas
  inputs are converted at node boundaries (CatBoost *could* use raw
  DataFrames for categoricals; v0.1 converts to numpy and therefore only
  supports numeric data -- categorical support is roadmap, see README).
- Torch adapters operate on ``torch.Tensor`` (float32, CPU by default).
- Batch dim is always first; single rows must still be 2D.
- Zero-copy where possible: ``torch.from_numpy`` shares memory,
  ``np.asarray`` avoids copies, ``DataFrame.to_numpy()`` is used once per
  boundary (pandas may copy for mixed dtypes -- unavoidable).

All conversions are explicit and validated; failures raise DataError with
the offending shape/dtype attached.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from .exceptions import DataError

ArrayLike = Any


def is_torch_tensor(x: Any) -> bool:
    return type(x).__module__.split(".")[0] == "torch" and hasattr(x, "detach")


def to_numpy(x: ArrayLike, *, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Convert numpy/pandas/torch/list input to a 2D ``np.ndarray``."""
    import pandas as pd

    if is_torch_tensor(x):
        try:
            arr = x.detach().cpu().numpy()
        except Exception as e:  # e.g. requires-grad edge cases
            raise DataError(f"Could not convert torch Tensor to numpy: {e}") from e
    elif isinstance(x, pd.DataFrame):
        arr = x.to_numpy()
    elif isinstance(x, pd.Series):
        arr = x.to_numpy()
    elif isinstance(x, np.ndarray):
        arr = x
    else:
        try:
            arr = np.asarray(x)
        except Exception as e:
            raise DataError(
                f"Unsupported input type {type(x).__name__}: cannot convert to array. "
                f"Pass numpy.ndarray, pandas.DataFrame/Series, torch.Tensor, or "
                f"array-like. Got error: {e}"
            ) from e
    if dtype is not None:
        try:
            arr = arr.astype(dtype, copy=False)
        except Exception as e:
            raise DataError(f"Could not cast array to {dtype}: {e}") from e
    return arr


def to_tensor(x: ArrayLike, *, dtype: str = "float32"):
    """Convert input to a ``torch.Tensor`` (float32 by default). Zero-copy
    from numpy via ``torch.from_numpy`` when dtypes align."""
    import torch

    if is_torch_tensor(x):
        t = x
    elif isinstance(x, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(x))
    else:
        arr = to_numpy(x)
        t = torch.from_numpy(np.ascontiguousarray(arr))
    want = getattr(torch, dtype)
    if t.dtype != want:
        t = t.to(want)
    return t


def ensure_2d(arr: np.ndarray, *, name: str = "input") -> np.ndarray:
    """Enforce 2D ``(n_samples, n_features)``. 1D ``(n,)`` is rejected with a
    helpful message (use ``as_column`` for predictions)."""
    arr = np.asanyarray(arr)
    if arr.ndim != 2:
        raise DataError(
            f"{name}: expected 2D array (n_samples, n_features), got shape "
            f"{arr.shape} (ndim={arr.ndim}). Reshape single rows to (1, d) and "
            f"single columns to (n, 1)."
        )
    if arr.shape[0] == 0:
        raise DataError(f"{name}: got 0 rows (shape {arr.shape}).")
    return arr


def as_column(arr: np.ndarray, *, name: str = "output") -> np.ndarray:
    """Reshape ``(n,)`` prediction vectors to ``(n, 1)`` for downstream
    feature use; pass through ``(n, k)`` matrices unchanged."""
    arr = np.asanyarray(arr)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    raise DataError(f"{name}: cannot use array with shape {arr.shape} as features.")


def check_row_count(*arrays: np.ndarray, name: str = "merge") -> int:
    """Validate all arrays share the same row count; return it."""
    counts = [np.asanyarray(a).shape[0] for a in arrays]
    if len(set(counts)) != 1:
        raise DataError(
            f"{name}: row-count mismatch {counts}. All branches merged together "
            f"must produce the same number of rows (same batch)."
        )
    return counts[0]


def infer_n_features(x: ArrayLike) -> int:
    arr = to_numpy(x)
    if arr.ndim == 1:
        return 1
    return int(arr.shape[1])


def target_to_numpy(y: ArrayLike) -> np.ndarray:
    """Targets may be 1D ``(n,)`` or 2D ``(n, k)``; anything else is an error."""
    import pandas as pd

    if isinstance(y, pd.Series):
        arr = y.to_numpy()
    elif is_torch_tensor(y):
        arr = y.detach().cpu().numpy()
    else:
        arr = np.asarray(y)
    if arr.ndim not in (1, 2):
        raise DataError(
            f"Target y must be 1D (n,) or 2D (n, k); got shape {arr.shape}."
        )
    if arr.shape[0] == 0:
        raise DataError("Target y has 0 rows.")
    return arr


def leaves_to_onehot(leaf_matrix: np.ndarray) -> np.ndarray:
    """Helper (roadmap): expand CatBoost/XGB ``apply()`` leaf indices into a
    numeric feature matrix via one-hot per tree. Not used by default in v0.1
    (nodes use predictions), but provided for future Transform-kind nodes."""
    leaf_matrix = np.asanyarray(leaf_matrix)
    if leaf_matrix.ndim != 2:
        raise DataError(f"leaf matrix must be 2D, got {leaf_matrix.shape}")
    parts = []
    for col in range(leaf_matrix.shape[1]):
        vals, inv = np.unique(leaf_matrix[:, col], return_inverse=True)
        oh = np.zeros((leaf_matrix.shape[0], len(vals)), dtype=np.float32)
        oh[np.arange(len(inv)), inv] = 1.0
        parts.append(oh)
    return np.concatenate(parts, axis=1)
