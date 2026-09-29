"""Explicit input/output contracts.

Critical design point: ``CatBoost: 12 -> 4`` is NOT ``nn.Linear(12, 4)``.
For tree models "outputs" means predictions of a specific kind:

- ``PREDICTION``: ``(n,)`` class labels or regression values (one estimator).
- ``PROBA`` / ``LOGITS``: ``(n, k)`` per-class probabilities/logits.
- ``TRANSFORM``: ``(n, k)`` transformed features (leaf indices, embeddings,
  or a bank of ``k`` estimators whose predictions become downstream features).
- ``FEATURES``: raw pass-through features (InputNode / Concat output).

Every Node declares what it consumes (FeatureSchema) and what it produces
(OutputSchema). Connections are validated at trace time (symbolic shapes) and
again at runtime (real data), before any execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TaskType(str, Enum):
    REGRESSION = "regression"
    BINARY = "binary"
    MULTICLASS = "multiclass"
    MULTIOUTPUT = "multioutput"
    FEATURES = "features"  # raw features, no supervised task
    UNSPECIFIED = "unspecified"


class OutputKind(str, Enum):
    FEATURES = "features"  # (n, d) feature matrix, e.g. Input or Concat output
    PREDICTION = "prediction"  # (n,) labels/values from a single estimator
    PROBA = "proba"  # (n, k) class probabilities
    LOGITS = "logits"  # (n, k) unnormalized scores
    TRANSFORM = "transform"  # (n, k) derived features (leaves, bank of estimators)


@dataclass(frozen=True)
class FeatureSchema:
    """What a node consumes: a 2D feature matrix with ``n_features`` columns."""

    n_features: Optional[int] = None  # None = any width accepted
    dtype: Optional[str] = None  # e.g. "float32"; None = any numeric dtype
    task: TaskType = TaskType.UNSPECIFIED

    def validate_against_output(self, out: "OutputSchema", *, node_name: str) -> None:
        from .exceptions import ShapeMismatchError, TaskMismatchError

        out_width = out.width()
        if self.n_features is not None and out_width is not None:
            if self.n_features != out_width:
                raise ShapeMismatchError(
                    f"Node '{node_name}' expects {self.n_features} features, "
                    f"but upstream node produces {out_width} "
                    f"(kind={out.kind.value}, dim={out.dim}). "
                    f"Fix the upstream output (e.g. use proba vs prediction, "
                    f"or insert a projection/merge) or change this node's "
                    f"expected input width."
                )
        # Task compatibility: FEATURES/UNSPECIFIED accept anything; otherwise
        # tasks must match unless one side is unspecified.
        if (
            self.task not in (TaskType.UNSPECIFIED, TaskType.FEATURES)
            and out.task not in (TaskType.UNSPECIFIED, TaskType.FEATURES)
            and self.task != out.task
        ):
            raise TaskMismatchError(
                f"Node '{node_name}' expects task '{self.task.value}' but "
                f"upstream produces task '{out.task.value}'. "
                f"Check output='prediction' vs 'proba' vs 'transform' settings."
            )


@dataclass(frozen=True)
class OutputSchema:
    """What a node produces."""

    kind: OutputKind = OutputKind.PREDICTION
    dim: Optional[int] = None  # columns out for matrix outputs; None = unknown until fit
    task: TaskType = TaskType.UNSPECIFIED
    dtype: Optional[str] = None

    def width(self) -> Optional[int]:
        """Width of this output when fed as features into a downstream node.

        - PREDICTION (n,) counts as width 1.
        - Matrix kinds count as ``dim`` (may be None = unknown until fit).
        """
        if self.kind == OutputKind.PREDICTION:
            return 1
        return self.dim

    def describe(self) -> str:
        return f"{self.kind.value}(dim={self.dim}, task={self.task.value})"
