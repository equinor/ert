from collections.abc import Sequence
from typing import Any

import pluggy
from pydantic import BaseModel, ValidationError

from everest.plugins.everest_plugin_manager import EverestPluginManager


class LazyEverestPluginManager:
    def __init__(self) -> None:
        self.everest_plugin_manager: EverestPluginManager | None = None

    @property
    def hook(self) -> pluggy.HookRelay:
        if self.everest_plugin_manager is None:
            self.everest_plugin_manager = EverestPluginManager()

        return self.everest_plugin_manager.hook


pm = LazyEverestPluginManager()


def collect_forward_model_schemas() -> dict[str, Any] | None:
    schemas = pm.hook.get_forward_models_schemas()
    if schemas:
        return schemas.pop()
    return {}


def lint_forward_model_job(job: str, args: Sequence[str]) -> list[str]:
    return pm.hook.lint_forward_model(job=job, args=args)


def validate_forward_model_step_arguments(forward_model_steps: list[str]) -> None:
    pm.hook.check_forward_model_arguments(forward_model_steps=forward_model_steps)


def parse_forward_model_file(path: str, schema: type[BaseModel], message: str) -> None:
    try:
        pm.hook.parse_forward_model_schema(path=path, schema=schema)
    except ValidationError as ve:
        raise ValueError(
            message.format(
                error="\n\t\t".join(
                    f"{error['loc'][0]}: {error['input']} -> {error['msg']}"
                    for error in ve.errors()
                )
            )
        ) from ve
    except ValueError as ve:
        raise ValueError(message.format(error=str(ve))) from ve
