import logging
import math
import sys
from typing import TYPE_CHECKING, Dict, List, Type, TypedDict, Union

from .parsing import ConfigValidationError

if sys.version_info < (3, 11):
    from enum import Enum

    class StrEnum(str, Enum):
        pass

else:
    from enum import StrEnum


logger = logging.getLogger(__name__)


if TYPE_CHECKING:

    class VariableInfo(TypedDict):
        type: Union[Type[float], Type[int]]
        min: float
        value: Union[float, int]
        max: float
        step: float
        labelname: str


DEFAULT_IES_MAX_STEPLENGTH = 0.60
DEFAULT_IES_MIN_STEPLENGTH = 0.30
DEFAULT_IES_DEC_STEPLENGTH = 2.50
DEFAULT_ENKF_TRUNCATION = 0.98
DEFAULT_IES_INVERSION = 0
DEFAULT_LOCALIZATION = False
# Default threshold is a function of ensemble size which is not available here.
DEFAULT_LOCALIZATION_CORRELATION_THRESHOLD = -1


def correlation_threshold(ensemble_size: int, user_defined_threshold: float) -> float:
    """Decides whether or not to use user-defined or default threshold.

    Default threshold taken from luo2022,
    Continuous Hyper-parameter OPtimization (CHOP) in an ensemble Kalman filter
    Section 2.3 - Localization in the CHOP problem
    """
    default_threshold = 3 / math.sqrt(ensemble_size)
    if user_defined_threshold == -1:
        return default_threshold

    return user_defined_threshold


class AnalysisMode(StrEnum):
    ITERATED_ENSEMBLE_SMOOTHER = "IES_ENKF"
    ENSEMBLE_SMOOTHER = "STD_ENKF"


def get_mode_variables(mode: AnalysisMode) -> Dict[str, "VariableInfo"]:
    es_variables: Dict[str, "VariableInfo"] = {
        "IES_INVERSION": {
            "type": int,
            "min": 0,
            "value": DEFAULT_IES_INVERSION,
            "max": 3,
            "step": 1,
            "labelname": "Inversion algorithm",
        },
        "ENKF_TRUNCATION": {
            "type": float,
            "min": -2.0,
            "value": DEFAULT_ENKF_TRUNCATION,
            "max": 1,
            "step": 0.01,
            "labelname": "Singular value truncation",
        },
        "LOCALIZATION": {
            "type": bool,
            "min": 0.0,
            "value": DEFAULT_LOCALIZATION,
            "max": 1.0,
            "step": 1.0,
            "labelname": "Adaptive localization",
        },
        "LOCALIZATION_CORRELATION_THRESHOLD": {
            "type": float,
            "min": 0.0,
            "value": DEFAULT_LOCALIZATION_CORRELATION_THRESHOLD,
            "max": 1.0,
            "step": 0.1,
            "labelname": "Adaptive localization correlation threshold",
        },
    }
    ies_variables: Dict[str, "VariableInfo"] = {
        "IES_MAX_STEPLENGTH": {
            "type": float,
            "min": 0.1,
            "value": DEFAULT_IES_MAX_STEPLENGTH,
            "max": 1.00,
            "step": 0.1,
            "labelname": "Gauss–Newton maximum steplength",
        },
        "IES_MIN_STEPLENGTH": {
            "type": float,
            "min": 0.1,
            "value": DEFAULT_IES_MIN_STEPLENGTH,
            "max": 1.00,
            "step": 0.1,
            "labelname": "Gauss–Newton minimum steplength",
        },
        "IES_DEC_STEPLENGTH": {
            "type": float,
            "min": 1.1,
            "value": DEFAULT_IES_DEC_STEPLENGTH,
            "max": 10.00,
            "step": 0.1,
            "labelname": "Gauss–Newton steplength decline",
        },
        **es_variables,
    }
    if mode == AnalysisMode.ENSEMBLE_SMOOTHER:
        return es_variables
    return ies_variables


class AnalysisModule:
    DEPRECATED_KEYS = ["USE_EE", "USE_GE"]
    TRUNC_ALTERNATE_KEYS = ["ENKF_NCOMP", "ENKF_SUBSPACE_DIMENSION"]
    SPECIAL_KEYS = ["INVERSION", *TRUNC_ALTERNATE_KEYS, *DEPRECATED_KEYS]

    def __init__(
        self,
        mode: AnalysisMode,
        name: str,
        variables: Dict[str, "VariableInfo"],
        iterable: bool,
    ):
        self.mode = mode
        self.name = name
        self.iterable = iterable
        self._variables = variables

    @classmethod
    def ens_smoother_module(cls, name: str = "STD_ENKF") -> "AnalysisModule":
        return cls(
            mode=AnalysisMode.ENSEMBLE_SMOOTHER,
            name=name,
            variables=get_mode_variables(AnalysisMode.ENSEMBLE_SMOOTHER),
            iterable=False,
        )

    @classmethod
    def iterated_ens_smoother_module(cls, name: str = "IES_ENKF") -> "AnalysisModule":
        return cls(
            mode=AnalysisMode.ITERATED_ENSEMBLE_SMOOTHER,
            name=name,
            variables=get_mode_variables(AnalysisMode.ITERATED_ENSEMBLE_SMOOTHER),
            iterable=True,
        )

    def get_variable_names(self) -> List[str]:
        return list(self._variables.keys())

    def get_variable_value(self, name: str) -> Union[int, float, bool]:
        if name in self._variables:
            return self._variables[name]["value"]
        raise ConfigValidationError(
            f"Variable {name!r} not found in module {self.name!r}"
        )

    def variable_value_dict(self) -> Dict[str, Union[float, int]]:
        return {name: var["value"] for name, var in self._variables.items()}

    def handle_special_key_set(
        self, var_name: str, value: Union[float, int, bool, str]
    ) -> None:
        if var_name in self.DEPRECATED_KEYS:
            logger.warning(
                f"The {var_name} key have been removed" f"use the INVERSION key instead"
            )
        elif var_name == "INVERSION":
            inversion_str_map = {
                "EXACT": 0,
                "SUBSPACE_EXACT_R": 1,
                "SUBSPACE_EE_R": 2,
                "SUBSPACE_RE": 3,
            }
            if value in inversion_str_map:
                assert isinstance(value, str)
                self.set_var("IES_INVERSION", inversion_str_map[value])
            else:
                logger.warning(
                    f"Unknown value {value} used for INVERSION key"
                    f"supported values are {list(inversion_str_map.keys())}"
                )
        elif var_name in self.TRUNC_ALTERNATE_KEYS:
            self.set_var("ENKF_TRUNCATION", value)

    def set_var(self, var_name: str, value: Union[float, int, bool, str]) -> None:
        if var_name in self.SPECIAL_KEYS:
            self.handle_special_key_set(var_name, value)
        elif var_name in self._variables:
            var = self._variables[var_name]

            if var["type"] is not bool:
                try:
                    new_value = var["type"](value)
                    if new_value > var["max"]:
                        var["value"] = var["max"]
                        logger.warning(
                            f"New value {new_value} for key"
                            f" {var_name} is out of [{var['min']}, {var['max']}] "
                            f"using max value {var['max']}"
                        )
                    elif new_value < var["min"]:
                        var["value"] = var["min"]
                        logger.warning(
                            f"New value {new_value} for key"
                            f" {var_name} is out of [{var['min']}, {var['max']}] "
                            f"using min value {var['min']}"
                        )
                    else:
                        var["value"] = new_value

                except ValueError as e:
                    raise ConfigValidationError(
                        f"Variable {var_name!r} with value {value!r} has "
                        f"incorrect type."
                        f" Expected type {var['type'].__name__!r} but received"
                        f" value {value!r} of type {type(value).__name__!r}"
                    ) from e
            else:
                if not isinstance(var["value"], bool):
                    raise ValueError(
                        f"Variable {var_name} expected type {var['type']}"
                        f" received value `{value}` of type `{type(value)}`"
                    )
                # When config is first read, `value` is a string
                # that's either "False" or "True",
                # but since bool("False") is True we need to convert it to bool.
                if not isinstance(value, bool):
                    value = str(value).lower() != "false"

                var["value"] = var["type"](value)
        else:
            raise ConfigValidationError(
                f"Variable {var_name!r} not found in {self.name!r} analysis module"
            )

    @property
    def inversion(self) -> int:
        return self.get_variable_value("IES_INVERSION")  # type: ignore

    @inversion.setter
    def inversion(self, value: int) -> None:
        self.set_var("IES_INVERSION", value)

    def get_truncation(self) -> float:
        return self.get_variable_value("ENKF_TRUNCATION")

    def localization(self) -> bool:
        return bool(self.get_variable_value("LOCALIZATION"))

    def localization_correlation_threshold(self, ensemble_size: int) -> float:
        return correlation_threshold(
            ensemble_size, self.get_variable_value("LOCALIZATION_CORRELATION_THRESHOLD")
        )

    def get_steplength(self, iteration_nr: int) -> float:
        """
        This is an implementation of Eq. (49), which calculates a suitable
        step length for the update step, from the book:
        Geir Evensen, Formulating the history matching problem with
        consistent error statistics, Computational Geosciences (2021) 25:945 –970

        Function not really used moved from C to keep the class interface consistent
        should be investigated for possible removal.
        """
        min_step_length = self.get_variable_value("IES_MIN_STEPLENGTH")
        max_step_length = self.get_variable_value("IES_MAX_STEPLENGTH")
        dec_step_length = self.get_variable_value("IES_DEC_STEPLENGTH")
        step_length = min_step_length + (max_step_length - min_step_length) * pow(
            2, -(iteration_nr - 1) / (dec_step_length - 1)
        )
        return step_length

    def __repr__(self) -> str:
        return f"AnalysisModule(name = {self.name})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AnalysisModule):
            return False

        if self.name != other.name:
            return False

        if self.mode != other.mode:
            return False

        if self._variables != other._variables:
            return False

        return True
