from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ert._c_wrappers.enkf.config.response_config import ResponseConfig

if TYPE_CHECKING:
    from typing import List, Optional

    from ecl.summary import EclSum


@dataclass
class SummaryConfig(ResponseConfig):
    input_file: str
    keys: List[str]
    refcase: Optional[EclSum] = None

    def __eq__(self, other):
        if (
            self.input_file != other.input_file
            or self.keys != other.keys
            or self.refcase.case != other.refcase.case
        ):
            return False
        return True
