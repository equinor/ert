from collections.abc import Callable
from functools import wraps
from typing import Any, ParamSpec

from .ert_plugin import CancelPluginException, ErtPlugin
from .ert_script import ErtScript
from .external_ert_script import ExternalErtScript
from .plugin_manager import (
    ErtPluginContext,
    ErtPluginManager,
    JobDoc,
    hook_implementation,
)
from .plugin_response import PluginMetadata, PluginResponse
from .workflow_config import ErtScriptWorkflow, WorkflowConfigs
from .workflow_fixtures import (
    HookedWorkflowFixtures,
    PostExperimentFixtures,
    PostSimulationFixtures,
    PostUpdateFixtures,
    PreExperimentFixtures,
    PreFirstUpdateFixtures,
    PreSimulationFixtures,
    PreUpdateFixtures,
    WorkflowFixtures,
    all_hooked_workflow_fixtures,
    fixtures_per_hook,
)

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
                    "ecl100_config_path",
                    "ecl300_config_path",
                    "flow_config_path",
                    "site_config_lines",
                    "activate_script",
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
    "ErtPluginContext",
    "ErtPluginManager",
    "ErtScript",
    "ErtScriptWorkflow",
    "ExternalErtScript",
    "HookedWorkflowFixtures",
    "JobDoc",
    "PostExperimentFixtures",
    "PostSimulationFixtures",
    "PostUpdateFixtures",
    "PreExperimentFixtures",
    "PreFirstUpdateFixtures",
    "PreSimulationFixtures",
    "PreUpdateFixtures",
    "WorkflowConfigs",
    "WorkflowFixtures",
    "all_hooked_workflow_fixtures",
    "fixtures_per_hook",
    "plugin",
]
