from typing import List, Optional

from pydantic import BaseModel, ValidationError

from ert3.config import PluginConfigManager
from ert3.config import ErtBaseModel, ErtPluginField


class _StagesConfig(BaseModel):
    class Config:
        validate_all = True
        validate_assignment = True
        extra = "forbid"
        allow_mutation = False
        arbitrary_types_allowed = True


class TransformationConfigBase(_StagesConfig):
    """Common config for all Transformations"""

    location: str


class FileTransformationConfig(TransformationConfigBase):
    mime: str = ""


class SummaryTransformationConfig(TransformationConfigBase):
    smry_keys: Optional[List[str]] = None


class DirectoryTransformationConfig(TransformationConfigBase):
    pass


class DummyInstance:
    """
    This is dummy instance that just stores the config.
    This would be the actual Transformations in the real case.
    """

    def __init__(self, config) -> None:
        self._config = config

    def __str__(self) -> str:
        return f"DummyInstance(config={self._config})"


def get_configs(pm):
    """
    We now need the create configs that use the ErtPluginField dynamically,
    as we would like to control the schema created based on the plugins we provide.
    This will allow us to specify a subset of plugins we want to have effect at runtime,
    such as only using configs from ert in tests.
    """

    class StageIO(ErtBaseModel, _StagesConfig, plugin_manager=pm):
        name: str
        transformation: ErtPluginField

    class Stage(BaseModel):
        input: StageIO

    return Stage


def register_plugins() -> PluginConfigManager:
    """
    This would normally be controlled by a module that calls the pluggy hooks,
    similar to ert_shared/plugins/plugin_manager.py.
    """
    pm = PluginConfigManager()
    pm.register_category(category="transformation")
    pm.register(
        name="file",
        category="transformation",
        config=FileTransformationConfig,
        factory=lambda x: DummyInstance(config=x),
    )
    pm.register(
        name="directory",
        category="transformation",
        config=DirectoryTransformationConfig,
        factory=lambda x: DummyInstance(config=x),
    )
    pm.register(
        name="summary",
        category="transformation",
        config=SummaryTransformationConfig,
        factory=lambda x: DummyInstance(config=x),
    )
    return pm


def main():
    pm = register_plugins()
    Stage = get_configs(pm=pm)

    stages_configs = [
        {
            "input": {
                "name": "test",
                "transformation": {"type": "file", "location": "test.json"},
            },
        },
        {
            "input": {
                "name": "test",
                "transformation": {"type": "directory", "location": "test/hei"},
            },
        },
        {
            "input": {
                "name": "test",
                "transformation": {"type": "file", "mime": "hei"},
            },
        },
        {
            "input": {
                "name": "test",
                "transformation": {
                    "type": "summary",
                    "location": "test.smry",
                    "smry_keys": ["WOPR", "FOPR"],
                },
            },
        },
    ]

    for config in stages_configs:
        print(f"Attempting to validate: {config}")
        try:
            stage = Stage(**config)
            print(stage)
            print(stage.input.get_transformation_instance())
        except ValidationError as e:
            print(f"Error: {e}")
        print()


"""
$ python test.py
Attempting to validate: {'input': {'name': 'test', 'transformation': {'type': 'file', 'location': 'test.json'}}}
input=StageIO(name='test', transformation=FullFileTransformationConfig(location='test.json', mime='', type='file'))
DummyInstance(config=location='test.json' mime='' type='file')

Attempting to validate: {'input': {'name': 'test', 'transformation': {'type': 'directory', 'location': 'test/hei'}}}
input=StageIO(name='test', transformation=FullDirectoryTransformationConfig(location='test/hei', type='directory'))
DummyInstance(config=location='test/hei' type='directory')

Attempting to validate: {'input': {'name': 'test', 'transformation': {'type': 'file', 'mime': 'hei'}}}
Error: 1 validation error for Stage
input -> transformation -> FullFileTransformationConfig -> location
  field required (type=value_error.missing)

Attempting to validate: {'input': {'name': 'test', 'transformation': {'type': 'summary', 'location': 'test.smry', 'smry_keys': ['WOPR', 'FOPR']}}}
input=StageIO(name='test', transformation=FullSummaryTransformationConfig(location='test.smry', smry_keys=['WOPR', 'FOPR'], type='summary'))
DummyInstance(config=location='test.smry' smry_keys=['WOPR', 'FOPR'] type='summary')
"""

if __name__ == "__main__":
    main()
