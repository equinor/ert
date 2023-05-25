from dataclasses import dataclass

from ert._c_wrappers.enkf.config.response_config import ResponseConfig


@dataclass
class SummaryConfig(ResponseConfig):
    ...
