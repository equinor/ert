from typing import List, Optional

from pydantic.dataclasses import dataclass


@dataclass
class Parameter:
    name: str
    index_list: Optional[List[int]] = None
