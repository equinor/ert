from typing import TypeVar

from pydantic import BaseModel, ValidationError

from ert.config import ConfigWarning
from everest.plugins.everest_plugin_manager import EverestPluginManager

pm = EverestPluginManager()
T = TypeVar("T", bound=BaseModel)


def collect_forward_model_schemas():
    schemas = pm.hook.get_forward_models_schemas()
    if schemas:
        return schemas.pop()
    return {}


def lint_forward_model_job(job: str, args) -> list[str]:
    return pm.hook.lint_forward_model(job=job, args=args)


def check_forward_model_objective(
    forward_model_steps: list[str], objectives: set[str]
) -> None:
    if not objectives or not forward_model_steps:
        return
    fm_outputs = pm.hook.custom_forward_model_outputs(
        forward_model_steps=forward_model_steps,
    )
    if fm_outputs is None:
        return
    unaccounted_objectives = objectives.difference(fm_outputs)
    if unaccounted_objectives:
        add_s = "s" if len(unaccounted_objectives) > 1 else ""
        ConfigWarning.warn(
            f"Warning: Forward model might not write the required output file{add_s}"
            f" for {sorted(unaccounted_objectives)}"
        )


def parse_forward_model_file(path: str, schema: type[T], message: str) -> T:
    try:
        res = pm.hook.parse_forward_model_schema(path=path, schema=schema)
        if res:
            res.pop()
        return res
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
