import sys
from typing import Any, Dict, Iterator, List, Union, Optional
from pydantic import (
    BaseModel,
    ValidationError,
    root_validator,
    validator,
)

import ert
import ert3

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


class _ParametersConfig(BaseModel):
    class Config:
        validate_all = True
        validate_assignment = True
        extra = "forbid"
        allow_mutation = False
        arbitrary_types_allowed = True


def _ensure_valid_name(name: str) -> str:
    if not name:
        raise ValueError("Names cannot be of zero length")

    if not all(c.isalpha() or c == "_" for c in name):
        raise ValueError(
            "Names are expected to only contain characters and `_`, was: {name}"
        )

    return name


class _GaussianInput(_ParametersConfig):
    mean: float
    std: float

    @validator("std")
    def _ensure_positive_std(cls, value):  # type: ignore
        if value is None:
            return None

        if value <= 0:
            raise ValueError(f"Expected positive std, was {value}")
        return value


class _UniformInput(_ParametersConfig):
    lower_bound: float
    upper_bound: float

    @root_validator
    def _ensure_lower_upper(cls, values):  # type: ignore
        low = values.get("lower_bound")
        up = values.get("upper_bound")

        if low is None or up is None:
            return values

        if low <= up:
            return values

        raise ValueError(
            f"Expected lower_bound ({low}) to be smaller than upper_bound ({up})"
        )


class _GaussianDistribution(_ParametersConfig):
    type: Literal["gaussian"]
    input: _GaussianInput


class _UniformDistribution(_ParametersConfig):
    type: Literal["uniform"]
    input: _UniformInput


class _VariablesConfig(_ParametersConfig):
    __root__: List[str]

    @validator("__root__")
    def _ensure_variables(cls, variables):  # type: ignore
        if len(variables) > 0:
            return variables

        raise ValueError("Parameter group cannot have no variables")

    @validator("__root__", each_item=True)
    def _ensure_valid_variable_names(cls, variable: Any) -> str:
        return _ensure_valid_name(variable)

    def __iter__(self) -> Iterator[str]:  # type: ignore
        return iter(self.__root__)

    def __getitem__(self, item: int) -> str:
        return self.__root__[item]

    def __len__(self) -> int:
        return len(self.__root__)


def _ensure_valid_size(size: Optional[int]) -> Optional[int]:
    if size is not None and size <= 0:
        raise ValueError("Size cannot be <= 0")

    return size


class _ParameterConfig(_ParametersConfig):
    name: str
    type: Literal["stochastic"]
    distribution: Union[_GaussianDistribution, _UniformDistribution]
    variables: Optional[_VariablesConfig] = None
    size: Optional[int] = None

    @root_validator
    def _ensure_variables_or_size(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        has_variables = values.get("variables")
        has_size = values.get("size")
        if has_variables and has_size:
            raise AssertionError("Parameter group cannot have both variables and size")
        if not has_variables and not has_size:
            raise AssertionError(
                "Parameter group cannot have neither variables nor size"
            )
        return values

    @validator("name")
    def _ensure_valid_group_name(cls, value: Any) -> str:
        return _ensure_valid_name(value)

    @validator("size")
    def _ensure_valid_group_size(cls, value: Any) -> Optional[int]:
        return _ensure_valid_size(value)

    def as_distribution(self) -> ert3.stats.Distribution:
        dist_config = self.distribution
        if self.variables is not None:
            index: Optional[ert.data.RecordIndex] = tuple(self.variables)
            size: Optional[int] = None
        else:
            size = self.size
            index = None

        if dist_config.type == "gaussian":
            assert dist_config.input.mean is not None
            assert dist_config.input.std is not None

            return ert3.stats.Gaussian(
                dist_config.input.mean,
                dist_config.input.std,
                index=index,
                size=size,
            )
        elif dist_config.type == "uniform":
            assert dist_config.input.lower_bound is not None
            assert dist_config.input.upper_bound is not None

            return ert3.stats.Uniform(
                dist_config.input.lower_bound,
                dist_config.input.upper_bound,
                index=index,
                size=size,
            )
        else:
            raise ValueError(f"Unknown distribution type: {dist_config.type}")


class ParametersConfig(_ParametersConfig):
    __root__: List[_ParameterConfig]

    def __iter__(self) -> Iterator[_ParameterConfig]:  # type: ignore
        return iter(self.__root__)

    def __getitem__(self, item: Union[int, str]) -> _ParameterConfig:
        if isinstance(item, int):
            return self.__root__[item]
        elif isinstance(item, str):
            for group in self:
                if group.name == item:
                    return group
            raise ValueError(f"No parameter group found named: {item}")
        raise TypeError(f"Item should be int or str, not {type(item)}")

    def __len__(self) -> int:
        return len(self.__root__)


def load_parameters_config(config: List[Dict[str, Any]]) -> ParametersConfig:
    try:
        return ParametersConfig.parse_obj(config)
    except ValidationError as err:
        raise ert.exceptions.ConfigValidationError(str(err), source="parameters")
