from dataclasses import dataclass
from typing import List, Optional

from ert._c_wrappers.enkf.row_scaling import RowScaling


@dataclass
class Parameter:
    name: str
    index_list: Optional[List[int]] = None


@dataclass
class RowScalingParameter:
    name: str
    row_scaling: RowScaling
    index_list: Optional[List[int]] = None
