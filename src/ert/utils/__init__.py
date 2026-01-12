import logging
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from functools import wraps
from typing import ParamSpec, TypeVar

from _ert.utils import file_safe_timestamp

P = ParamSpec("P")
R = TypeVar("R")


def log_duration(
    logger: logging.Logger,
    logging_level: int = logging.DEBUG,
    custom_name: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    This is a decorator that logs the time it takes for a function to execute
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            t = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_time = time.perf_counter() - t
            name = custom_name or f"{func.__name__}()"
            logger.log(logging_level, f"{name} time_used={elapsed_time:.4f}s")
            return result

        return wrapper

    return decorator


def makedirs_if_needed(path: str, roll_if_exists: bool = False) -> None:
    if os.path.isdir(path):
        if not roll_if_exists:
            return
        _roll_dir(path)  # exists and should be rolled
    os.makedirs(path)


def _roll_dir(old_name: str) -> None:
    old_name = os.path.realpath(old_name)
    timestamp = file_safe_timestamp(datetime.now(UTC).isoformat())
    new_name = f"{old_name}__{timestamp}"
    os.rename(old_name, new_name)
    logging.getLogger().info(f"renamed {old_name} to {new_name}")
