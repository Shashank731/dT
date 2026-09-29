"""Exception hierarchy with clear, actionable messages."""


class DTError(Exception):
    """Base class for all dT errors."""


class SchemaMismatchError(DTError):
    """Raised when an upstream output schema cannot satisfy a downstream input schema."""


class ShapeMismatchError(SchemaMismatchError):
    """Feature-count / output-dim mismatch, e.g. 'expects 4 features but got 7'."""


class DTypeMismatchError(SchemaMismatchError):
    """Dtype contract violation."""


class TaskMismatchError(SchemaMismatchError):
    """E.g. feeding regression output into a node expecting multiclass proba."""


class UnfittedNodeError(DTError):
    """Raised when predict() is called on a node/graph with unfitted estimators."""


class CycleError(DTError):
    """Raised when the traced computation contains a cycle (not a DAG)."""


class SerializationError(DTError):
    """Raised when save/load fails."""


class DataError(DTError):
    """Raised for bad input data (wrong rank, empty, ragged, ...)."""
