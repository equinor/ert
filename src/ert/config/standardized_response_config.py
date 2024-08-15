import dataclasses
from functools import cached_property
from typing import Any, Dict, List, Literal, Optional

import xarray as xr

from .response_config import ResponseConfig
from .responses_index import responses_index


@dataclasses.dataclass
class ResponseConfigArgs:
    name: str
    input_file: str
    keys: List[str]
    kwargs: Dict[str, Any]

    def __post_init__(self):
        self.keys.sort()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "input_file": self.input_file,
            "kwargs": self.kwargs,
            "keys": self.keys,
        }


@dataclasses.dataclass
class StandardResponseConfig:
    ert_kind: str
    args_per_instance: List[ResponseConfigArgs] = dataclasses.field(
        default_factory=list
    )

    @staticmethod
    def create(source_configs: List[ResponseConfig]) -> "StandardResponseConfig":
        first_config = source_configs[0]

        _ert_kind = first_config.__class__.__name__
        response_args_list = [
            ResponseConfigArgs(
                name=config.name,
                input_file=config.input_file,
                kwargs=config.kwargs,
                keys=list(set(config.keys or {config.name})),
            )
            for config in source_configs
        ]

        return StandardResponseConfig(
            ert_kind=_ert_kind,
            args_per_instance=response_args_list,
        )

    @staticmethod
    def standardize_configs(
        response_configs: Optional[List[ResponseConfig]],
    ) -> List["StandardResponseConfig"]:
        if response_configs is None:
            return []

        configs_by_response_type: Dict[str, List[ResponseConfig]] = {}
        for config in response_configs:
            response_type = config.response_type

            if response_type not in configs_by_response_type:
                configs_by_response_type[response_type] = []

            configs_by_response_type[response_type].append(config)

        standard_format_configs: List[StandardResponseConfig] = []
        for configs in configs_by_response_type.values():
            standard_config = StandardResponseConfig.create(source_configs=configs)
            standard_format_configs.append(standard_config)

        standard_format_configs.sort(key=lambda x: x.ert_kind)

        return standard_format_configs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "_ert_kind": self.ert_kind,
            "args_per_instance": [x.to_dict() for x in self.args_per_instance],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StandardResponseConfig":
        return cls(
            ert_kind=d["_ert_kind"],
            args_per_instance=[
                ResponseConfigArgs(**args) for args in d["args_per_instance"]
            ],
        )

    @property
    def keys(self) -> List[str]:
        if self.first_config_instance.cardinality == "one_per_key":
            return sorted([config.name for config in self.all_config_instances])

        return self.first_config_instance.keys

    @property
    def cardinality(self) -> Literal["one_per_key", "one_per_realization"]:
        return self.first_config_instance.cardinality

    @property
    def response_type(self) -> str:
        return self.first_config_instance.response_type

    @property
    def input_files(self) -> List[str]:
        return [config.input_file for config in self.all_config_instances]

    @property
    def first_config_instance(self) -> ResponseConfig:
        return self.all_config_instances[0]

    @cached_property
    def all_config_instances(self) -> List[ResponseConfig]:
        configs = []
        config_cls = responses_index[self.ert_kind]

        for args in self.args_per_instance:
            instance = config_cls(
                name=args.name,
                input_file=args.input_file,
                keys=args.keys,
                kwargs=args.kwargs,
            )
            configs.append(instance)

        return configs

    @staticmethod
    def _all_keys(source_configs: List[ResponseConfig]) -> List[str]:
        if len(source_configs) == 0:
            return []

        first_config = source_configs[0]
        if first_config.cardinality == "one_per_key":
            return [config.name for config in source_configs]

        return first_config.keys

    @cached_property
    def config_instance(self) -> ResponseConfig:
        return responses_index[self.ert_kind](**self.to_dict())

    def read_from_file(self, run_path: str, iens: int) -> xr.Dataset:
        if self.first_config_instance.cardinality == "one_per_key":
            datasets = []
            for config in self.all_config_instances:
                ds = config.read_from_file(run_path, iens)
                datasets.append(ds.expand_dims(name=[config.name]))

            return xr.concat(datasets, dim="name")

        return self.first_config_instance.read_from_file(run_path, iens)
