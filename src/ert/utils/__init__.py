import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

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
