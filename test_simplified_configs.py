from typing import List, Optional

from pydantic import BaseModel, ValidationError, create_model

from ert3.config import ConfigPluginRegistry






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
