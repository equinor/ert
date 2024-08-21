from functools import wraps
from typing import Callable

from typing_extensions import Any, ParamSpec

from .plugin_manager import (
    ErtPluginContext,
    ErtPluginManager,
    JobDoc,
    hook_implementation,
)
from .plugin_response import PluginMetadata, PluginResponse

P = ParamSpec("P")


def plugin(name: str) -> Callable[[Callable[P, Any]], Callable[P, Any]]:
    def wrapper(func: Callable[P, Any]) -> Callable[P, Any]:
        @wraps(func)
        def inner(*args: P.args, **kwargs: P.kwargs) -> Any:
            res = func(*args, **kwargs)
            if (
                func.__name__
                in [
                    "installable_jobs",
                    "job_documentation",
                    "installable_workflow_jobs",
                    "help_links",
                    "installable_forward_model_steps",
                    "ecl100_config_path",
                    "ecl300_config_path",
                    "flow_config_path",
                    "help_links",
                    "site_config_lines",
                ]
                and res is not None
            ):
                return PluginResponse(
                    res,
                    PluginMetadata(name, func.__name__),
                )
            return res

        return hook_implementation(inner, specname=func.__name__)

    return wrapper


__all__ = ["ErtPluginContext", "ErtPluginManager", "JobDoc", "plugin"]
