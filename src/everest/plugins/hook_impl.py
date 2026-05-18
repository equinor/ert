from collections.abc import Sequence

from pydantic import BaseModel

from everest.plugins import hookimpl


@hookimpl
def lint_forward_model(job: str, args: Sequence[str]) -> None:
    return None


@hookimpl
def parse_forward_model_schema(path: str, schema: type[BaseModel]) -> None:
    return None


@hookimpl
def get_forward_models_schemas() -> None:
    return None


@hookimpl
def installable_workflow_jobs() -> None:
    return None


@hookimpl
def get_forward_model_documentations() -> None:
    return None
