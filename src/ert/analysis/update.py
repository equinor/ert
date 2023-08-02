from typing import List, Optional

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from .row_scaling import RowScaling


@dataclass
class Parameter:
    name: str
    index_list: Optional[List[int]] = None


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class RowScalingParameter:
    name: str
    row_scaling: RowScaling
    index_list: Optional[List[int]] = None
