"""Typed persistence failures exposed to services and jobs."""


class PersistenceError(RuntimeError):
    """Base class for local persistence failures."""


class RecordNotFoundError(PersistenceError):
    """Raised when a requested persisted identity does not exist."""


class ImmutableRecordError(PersistenceError):
    """Raised when an immutable revision or terminal run would change."""


class InvalidRunTransitionError(PersistenceError):
    """Raised when an eval run state transition is not allowed."""


class PersistenceValidationError(PersistenceError):
    """Raised when related records do not form a valid persisted graph."""
