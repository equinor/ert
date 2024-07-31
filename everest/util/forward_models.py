from itertools import chain
from typing import List, Type, TypeVar

from pydantic import BaseModel, ValidationError

from everest.plugins.hook_manager import EverestPluginManager

pm = EverestPluginManager()
T = TypeVar("T", bound=BaseModel)


def collect_forward_models():
    return chain.from_iterable(pm.hook.get_forward_models())


def collect_forward_model_schemas():
    schemas = pm.hook.get_forward_models_schemas()
    if schemas:
        return schemas.pop()
    return {}


def lint_forward_model_job(job: str, args) -> List[str]:
    return pm.hook.lint_forward_model(job=job, args=args)


def parse_forward_model_file(path: str, schema: Type[T], message: str) -> T:
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
