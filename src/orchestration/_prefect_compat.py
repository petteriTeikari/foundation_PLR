"""Prefect compatibility layer for optional dependency.

Provides task/flow/get_run_logger that work whether Prefect is
installed or not. Uses importlib to avoid static-analysis alias
resolution issues with griffe/mkdocstrings.

When Prefect is available and PREFECT_DISABLED is not set, returns
real Prefect decorators. Otherwise, returns no-op decorators.
"""

import importlib
import os
from typing import Any, Callable, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

USE_PREFECT = os.environ.get("PREFECT_DISABLED", "").lower() not in ("1", "true", "yes")
PREFECT_AVAILABLE = False


def _noop_task(fn: Optional[F] = None, **kwargs: Any) -> Any:
    """No-op task decorator when Prefect is not available."""
    if fn is None:
        return lambda f: f
    return fn


def _noop_flow(fn: Optional[F] = None, **kwargs: Any) -> Any:
    """No-op flow decorator when Prefect is not available."""
    if fn is None:
        return lambda f: f
    return fn


def _noop_get_run_logger() -> Any:
    """No-op get_run_logger when Prefect is not available."""
    import logging

    return logging.getLogger("prefect.compat")


task: Any = _noop_task
flow: Any = _noop_flow
get_run_logger: Any = _noop_get_run_logger

if USE_PREFECT:
    try:
        _prefect = importlib.import_module("prefect")
        task = _prefect.task
        flow = _prefect.flow
        get_run_logger = _prefect.get_run_logger
        PREFECT_AVAILABLE = True
    except ImportError:
        pass
