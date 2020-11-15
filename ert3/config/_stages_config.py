import importlib
from typing import List, Callable, Optional
from pydantic import validator, FilePath, BaseModel


def _import_from(path):
    try:
        module, func = path.split(":")
    except ValueError as err:
        raise (
            ValueError(
                f"Malformed script name, must be: some.module:function_name, was {path}"
            )
        ) from err
    except AttributeError as err:
        raise ValueError(f"Script must be type <str> is: {type(path)}") from err
    try:
        module = importlib.import_module(module)
    except ModuleNotFoundError as err:
        raise ValueError(f"No module named: {module}") from err
    try:
        func = getattr(module, func)
    except AttributeError as err:
        raise ValueError(f"No function named: {func}") from err
    return func


class _StagesConfig(BaseModel):
    validate_all = True
    validate_assignment = True
    extra = "forbid"
    allow_mutation = False
    arbitrary_types_allowed = True


class InputRecord(_StagesConfig):
    record: str
    location: str


class OutputRecord(_StagesConfig):
    record: str
    location: str


class Step(_StagesConfig):
    name: str
    script: List[Callable]
    input: List[InputRecord]
    output: List[OutputRecord]
    environment: str = None

    @validator("script", pre=True)
    def function_is_valid(cls, scripts: str) -> List[Callable]:
        return [_import_from(method) for method in scripts]


class StagesConfig(BaseModel):
    __root__: List[Step]

    def step_from_key(self, key: str) -> Optional[Step]:
        return next((step for step in self if step.name == key), None)

    def __iter__(self):
        return iter(self.__root__)

    def __getitem__(self, item):
        return self.__root__[item]

    def __len__(self):
        return len(self.__root__)


def load_stages_config(config_dict):
    return StagesConfig.parse_obj(config_dict)
