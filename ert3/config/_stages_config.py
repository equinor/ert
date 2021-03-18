import importlib
import os
import sys
from typing import List, Callable, Optional
from pydantic import root_validator, validator, FilePath, BaseModel, ValidationError

import ert3

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


def _import_from(path):
    if ":" not in path:
        raise ValueError("Function should be defined as module:function")
    module_str, func = path.split(":")
    module = importlib.import_module(module_str)
    try:
        func = getattr(module, func)
    except AttributeError:
        raise ImportError(name=func, path=module_str)
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


class TransportableCommand(_StagesConfig):
    name: str
    location: FilePath

    @validator("location")
    def is_executable(cls, location):
        if not os.access(location, os.X_OK):
            raise ValueError(f"{location} is not executable")
        return location


class Step(_StagesConfig):
    name: str
    type: Literal["unix", "function"]
    script: Optional[List[str]] = []
    input: List[InputRecord]
    output: List[OutputRecord]
    transportable_commands: Optional[List[TransportableCommand]] = []
    function: Optional[Callable]

    @root_validator
    def check_defined(cls, step):
        cmd_names = [cmd.name for cmd in step.get("transportable_commands", [])]
        script_lines = step.get("script", [])
        if step.get("type") == "function":
            if cmd_names:
                raise ValueError("Commands defined for a function stage")
            if script_lines:
                raise ValueError("Scripts defined for a function stage")
            if not step.get("function"):
                raise ValueError("No function defined")
        elif step.get("type") == "unix":
            if step.get("function"):
                raise ValueError("Function defined for unix step")

        for script in script_lines:
            line_cmd = script.split()[0]
            if line_cmd not in cmd_names:
                raise ValueError("{} is not a known command".format(line_cmd))
        return step

    @validator("function", pre=True)
    def function_is_valid(cls, function: str) -> Callable:
        return _import_from(function)


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
    try:
        return StagesConfig.parse_obj(config_dict)
    except ValidationError as err:
        raise ert3.exceptions.ConfigValidationError(str(err), source="stages")
