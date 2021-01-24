import importlib
import os
from typing import List, Callable, Optional
from pydantic import root_validator, validator, FilePath, BaseModel


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
    script: List[str]
    input: List[InputRecord]
    output: List[OutputRecord]
    transportable_commands: List[TransportableCommand]

    @root_validator
    def command_defined(cls, step):
        valid_cmds = [cmd.name for cmd in step.get("transportable_commands", [])]
        for script_line in step.get("script", []):
            line_cmd = script_line.split()[0]
            if line_cmd not in valid_cmds:
                raise ValueError("{} is not a known command".format(line_cmd))
        return step


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
