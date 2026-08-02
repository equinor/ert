from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from everest.plugins import hookspec


@hookspec(firstresult=True)
def lint_forward_model(job: str, args: Sequence[str]) -> list[str]:  # type: ignore[empty-body]
    """
    Validate arguments for a forward model before it is run.

    :param job: Name of the forward model job.
    :param args: Command line arguments passed to the job.
    :return: A list of linting error messages, or ``None`` when the plugin does
        not handle the job.
    """


@hookspec
def get_forward_models_schemas() -> dict[str, dict[str, type[BaseModel]]]:  # type: ignore[empty-body]
    """
    Return pydantic schemas for forward model configuration files.

    The outer dictionary maps forward model names to their schema
    definitions. Each schema definition maps an argument name to the model used
    to parse that argument's configuration file, for example
    ``{"add_template": {"-c/--config": WellModelConfig}}``.
    """


@hookspec
def parse_forward_model_schema(path: str, schema: type[BaseModel]) -> BaseModel:  # type: ignore[empty-body]
    """
    Parse a forward model configuration file.

    :param path: Path to the configuration file supplied to the forward model.
    :param schema: Pydantic model type that should parse the file.
    :return: Parsed schema instance, or ``None`` when the plugin does not parse
        this schema type.
    """


@hookspec
def installable_workflow_jobs() -> dict[str, Any]:  # type: ignore[empty-body]
    """
    Return workflow jobs provided by the plugin.

    :return: Dictionary with workflow job names as keys and job configuration
        paths or definitions as values.
    """


@hookspec
def get_forward_model_documentations() -> dict[str, Any]:  # type: ignore[empty-body]
    """
    Return documentation metadata for plugin-provided forward models.

    :return: Dictionary keyed by forward model name. Values are consumed by the
        EVEREST documentation helpers.
    """


@hookspec()
def check_forward_model_arguments(forward_model_steps: list[str]) -> None:
    """
    Check whether configured forward model steps use valid arguments.

    :param forward_model_steps: Forward model step strings from the EVEREST
        configuration.
    :raises ValueError: If a step contains invalid arguments.
    """
