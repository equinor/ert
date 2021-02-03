import importlib
import os
from typing import List, Callable, Optional, Union
from pydantic import root_validator, validator, FilePath, BaseModel

def _import_from(path):
    try:
        module, func = path.split(":")
    except ValueError as err:
        return path
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


class TransportableCommand(_StagesConfig):
    name: str
    location: FilePath

    @validator("location")
    def is_executable(cls, location):
        if not os.access(location, os.X_OK):
            raise ValueError(f"{location} is not executable")
        return location

class TransportableFunction(_StagesConfig):
    name: str
    args: str


class Step(_StagesConfig):
    name: str
    script: List[Union[str, Callable]]
    input: Optional[List[InputRecord]] = []
    output: List[OutputRecord]
    transportable_commands: Optional[List[TransportableCommand]] = []
    transportable_functions: Optional[List[TransportableFunction]] = []

    def outputs(self):
        return [out.location for out in self.output]

    def inputs(self):
        return [input.location for input in self.input]

    def command_location(self, name):
        return next(cmd.location for cmd in self.transportable_commands if cmd.name == name)

    def function_args(self, name):
        return next(fun.args for fun in self.transportable_functions if fun.name == name)

    def command_scripts(self):
        if self.transportable_commands:
            return [cmd.location for cmd in self.transportable_commands]
        return []

    @root_validator
    def check_defined(cls, step):
        cmd_names = [cmd.name for cmd in step.get("transportable_commands", [])]
        fun_names = [fun.name for fun in step.get("transportable_functions", [])]
        for script in step.get("script", []):
            if isinstance(script, Callable):
                if script.__name__ not in fun_names:
                    raise ValueError("{} is not a known function".format(line_cmd))
            else:
                line_cmd = script.split()[0]
                if line_cmd not in cmd_names:
                    raise ValueError("{} is not a known command".format(line_cmd))
        return step

    @validator("script", pre=True)
    def function_is_valid(cls, scripts: str) -> List[Callable]:
        return [_import_from(element) for element in scripts]

class StagesConfig(BaseModel):
    __root__: List[Step]

    def step_from_key(self, key: str) -> Optional[Step]:
        return next((step for step in self if step.name == key), None)

    def get_prefect_stages(self, stage_names) -> List[dict]:
        config_stage = [self.step_from_key(key) for key in stage_names]
        prefect_stages = []
        for stage in config_stage:
            command_scripts = stage.command_scripts()
            jobs = []
            for script in stage.script:
                if isinstance(script, Callable):
                    jobs.append({
                        "name" : script.__name__,
                        "executable": script,
                        "args": stage.function_args(script.__name__)
                    })
                else:
                    name, *args = script.split()
                    jobs.append({
                        "name": name,
                        "executable": stage.command_location(name),
                        "args": tuple(args),
                    })

            prefect_stages.append({
                "name": stage.name,
                "steps": [
                    {
                        "name": stage.name + "-only_step",
                        "resources": command_scripts,
                        "inputs": [],
                        "outputs": stage.outputs(),
                        "jobs": jobs,
                    }
                ]
             })
        return prefect_stages
    def __iter__(self):
        return iter(self.__root__)

    def __getitem__(self, item):
        return self.__root__[item]

    def __len__(self):
        return len(self.__root__)


def load_stages_config(config_dict):
    return StagesConfig.parse_obj(config_dict)
