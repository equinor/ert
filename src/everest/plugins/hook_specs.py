from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from everest.plugins import hookspec


@hookspec(firstresult=True)
def visualize_data(api: Any) -> None:
    """
    :param :EverestAPI instance
    """


@hookspec(firstresult=True)
def install_job_directories() -> list[str]:  # type: ignore[empty-body]
    """
    :return: List default site config of lines to
    :rtype: List of strings
    """


@hookspec(firstresult=True)
def lint_forward_model(job: str, args: Sequence[str]) -> list[str]:  # type: ignore[empty-body]
    """
    Return a error string, if forward model job failed to lint.
    """


@hookspec
def get_forward_models_schemas() -> dict[str, dict[str, type[BaseModel]]]:  # type: ignore[empty-body]
    """
    Return a dictionary of forward model names and its associated: schemas.
    Example {"add_template": {"-c/--config": WellModelConfig}, ...}
    """


@hookspec
def parse_forward_model_schema(path: str, schema: type[BaseModel]) -> BaseModel:  # type: ignore[empty-body]
    """
    Given a path and schema type, this hook will parse the file.
    """


@hookspec
def installable_workflow_jobs() -> dict[str, Any]:  # type: ignore[empty-body]
    """
    :return: dict with workflow job names as keys and path to config as value
    :rtype: PluginResponse with data as dict[str,str]
    """


@hookspec
def get_forward_model_documentations() -> dict[str, Any]:  # type: ignore[empty-body]
    """ """


@hookspec()
def check_forward_model_arguments(forward_model_steps: list[str]) -> None:
    """
    Check if the given arguments given to the forward model steps are valid
    """
