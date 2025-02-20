from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from everest.plugins import hookimpl


@hookimpl
def visualize_data(api: Any) -> None:
    print("No visualization plugin installed!")


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
