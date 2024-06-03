import dataclasses
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import xarray as xr

from ert.config.commons import Refcase
from ert.config.parameter_config import CustomDict
from ert.config.parsing import ContextString, ContextValue
from ert.config.responses.observation_vector import ObsVector

from ..parsing.config_errors import ConfigValidationError


@dataclasses.dataclass
class ObsArgs:
    obs_name: str
    refcase: Optional[Refcase]
    values: Any
    std_cutoff: Any
    history: Optional[Any]
    obs_time_list: Any
    config_for_response: Optional["ResponseConfig"] = None


_PATTERN = re.compile(r"(<[^>]+>)=([^,]+?)(?=,|\)$)")


class ObservationConfig:
    src: str
    obs_name: str
    line_from_ert_config: ContextString
    response_name: Optional[str] = None
    response_type: Optional[str] = None

    def __init__(self, line_from_ert_config: ContextString):
        kwargs = ResponseConfigWithLifecycleHooks.parse_kwargs_from_config_list(
            line_from_ert_config
        )

        response_name = kwargs.get("<RESPONSE_NAME>")
        response_type = kwargs.get("<RESPONSE_TYPE>")
        obs_name = kwargs.get("<OBS_NAME>", f"obs{line_from_ert_config}")

        # Expect <SRC> always
        if "<SRC>" not in kwargs:
            raise ConfigValidationError(
                "Observation must have <SRC> keyword argument to specify the"
                "source of the observation."
            )

        self.src = kwargs["<SRC>"]
        self.obs_name = obs_name
        self.line_from_ert_config = line_from_ert_config
        self.response_type = response_type
        self.response_name = response_name


class ResponseConfigWithLifecycleHooks(ABC):
    line_from_ert_config: List[ContextValue]

    def __init__(
        self,
        line_from_ert_config: List[ContextValue],
    ):
        self.line_from_ert_config = line_from_ert_config

    @property
    def src(self):
        if len(self.line_from_ert_config) == 1:
            kwargs = self.parse_kwargs_from_config_list(self.line_from_ert_config[0])
            if "<SRC>" not in kwargs:
                raise ConfigValidationError(
                    f"Response of type {self.response_type()} must "
                    f"have <SRC> keyword argument to specify the name of the file"
                    f"it should be read from."
                )

            return kwargs["<SRC>"]

        raise ConfigValidationError(
            f"Response {self.name} expected args in format (<K>=V,...)"
        )

    @property
    def name(self):
        if len(self.line_from_ert_config) == 1:
            kwargs = self.parse_kwargs_from_config_list(self.line_from_ert_config[0])
            return kwargs.get(
                "<NAME>",
                f"response{self.line_from_ert_config[0]}",
            )

    @classmethod
    def parse_kwargs_from_config_list(cls, config_list: ContextValue) -> Dict[str, str]:
        return {m[1]: m[2] for m in _PATTERN.finditer(config_list)}

    @classmethod
    @abstractmethod
    def response_type(cls) -> str:
        """Denotes the name for the entire response type. Not to be confused
        with the name of the specific response. This would for example be
        GEN_DATA, whilst WOPR:OP1 would be the name of an instance of this class
        """

    @classmethod
    @abstractmethod
    def ert_config_response_keyword(cls) -> str:
        """Denotes the keyword to be used in the ert config to give responses
            of the implemented type. For example CSV_RESPONSE.
        :return: The ert config keyword for specifying responses for this type.
        """

    @classmethod
    @abstractmethod
    def ert_config_observation_keyword(cls) -> str:
        """Denotes the keyword to be used in the ert config to give observations
            on responses of this type. For example CSV_OBSERVATION.
        :return: The ert config keyword for specifying observations
                 on this response type.
        """

    @abstractmethod
    def parse_response_from_config(
        self, response_kwargs_from_ert_config: Dict[str, str]
    ) -> None:
        """
        Parses the response given the keyword arguments specified in the config.
        For example, if the ert config kwargs is
        (<A>=2,<B>=heh), then response_kwargs_from_ert_config will be
        {"<A>": "2", "<B>": "heh"}
        """

    @abstractmethod
    def parse_observation_from_config(
        self, observation_kwargs_from_ert_config: Dict[str, str]
    ) -> xr.Dataset:
        """
        Parses the observation given the keyword arguments specified
        for the observation entry in the config. For example, if line 5 in the
        ert config is CSV_OBSERVATION(<SRC>="22.txt"), the kwargs will be
        {"<SRC>": "22.txt", "OBS_NAME_DEFAULT": "Observation@line:5"}
        The "OBS_NAME_DEFAULT" is only meant to be used if the <SRC> entry itself
        does not contain a list of observation names.
        """
        pass

    @abstractmethod
    def parse_response_from_runpath(self, run_path: str) -> xr.Dataset:
        pass


@dataclasses.dataclass
class ResponseConfig(ABC):
    name: str

    @staticmethod
    @abstractmethod
    def parse_observation(args: ObsArgs) -> Dict[str, ObsVector]: ...

    @abstractmethod
    def read_from_file(self, run_path: str, iens: int) -> xr.Dataset: ...

    def to_dict(self) -> Dict[str, Any]:
        data = dataclasses.asdict(self, dict_factory=CustomDict)
        data["_ert_kind"] = self.__class__.__name__
        return data
