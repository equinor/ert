import logging
from collections.abc import Sequence
from typing import Any

from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic import BaseModel

from everest.plugins import hookspec


@hookspec(firstresult=True)
def visualize_data(api: Any) -> None:
    """
    :param :EverestAPI instance
    """


@hookspec(firstresult=True)
def default_site_config_lines() -> list[str]:  # type: ignore[empty-body]
    """
    :return: List default site config of lines to
    :rtype: List of strings
    """


@hookspec(firstresult=True)
def install_job_directories() -> list[str]:  # type: ignore[empty-body]
    """
    :return: List default site config of lines to
    :rtype: List of strings
    """


@hookspec()
def site_config_lines() -> list[str]:  # type: ignore[empty-body]
    """
    :return: List of lines to append to site config file
    :rtype: PluginResponse with data as list[str]
    """


@hookspec(firstresult=True)
def ecl100_config_path() -> str:  # type: ignore[empty-body]
    """
    :return: Path to ecl100 config file
    :rtype: PluginResponse with data as str
    """


@hookspec(firstresult=True)
def ecl300_config_path() -> str:  # type: ignore[empty-body]
    """
    :return: Path to ecl300 config file
    :rtype: PluginResponse with data as str
    """


@hookspec(firstresult=True)
def flow_config_path() -> str:  # type: ignore[empty-body]
    """
    :return: Path to flow config file
    :rtype: PluginResponse with data as str
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
def add_log_handle_to_root() -> logging.Handler:  # type: ignore[empty-body]
    """
    Create a log handle which will be added to the root logger
    in the main entry point.
    :return: A log handle that will be added to the root logger
    :rtype: logging.Handler
    """


@hookspec
def add_span_processor() -> BatchSpanProcessor:  # type: ignore
    """
    Create a BatchSpanProcessor which will be added to the trace provider
    in ert.

    :return: A BatchSpanProcessor that will be added to the trace provider in everest
    """


@hookspec
def get_forward_model_documentations() -> dict[str, Any]:  # type: ignore[empty-body]
    """ """


@hookspec()
def check_forward_model_arguments(forward_model_steps: list[str]) -> None:
    """
    Check if the given arguments given to the forward model steps are valid
    """
