from collections.abc import Callable
from functools import wraps
from typing import Any, ParamSpec

from ert.config.ert_plugin import CancelPluginException, ErtPlugin

from .plugin_manager import (
    ErtPluginManager,
    ErtRuntimePlugins,
    JobDoc,
    get_site_plugins,
    hook_implementation,
    setup_site_logging,
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
                in {
                    "installable_jobs",
                    "job_documentation",
                    "installable_workflow_jobs",
                    "help_links",
                    "installable_forward_model_steps",
                    "forward_model_configuration",
                    "site_configurations",
                }
                and res is not None
            ):
                return PluginResponse(
                    res,
                    PluginMetadata(name, func.__name__),
                )
            return res

        return hook_implementation(inner, specname=func.__name__)

    return wrapper


__all__ = [
    "CancelPluginException",
    "ErtPlugin",
    "ErtPluginManager",
    "ErtRuntimePlugins",
    "JobDoc",
    "get_site_plugins",
    "plugin",
    "setup_site_logging",
]
