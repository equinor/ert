import enum
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Set, Tuple

from numpy import float32, float64
from numpy.typing import NDArray

class Kind(enum.Enum):
    FIELD = 104
    GEN_KW = 107
    SUMMARY = 110
    GEN_DATA = 113
    SURFACE = 114
    EXT_PARAM = 116

@dataclass
class Block:
    kind: Kind
    name: str
    report_step: int
    realization_index: int

class DataFile:
    def __init__(self, path: Path) -> None: ...
    def blocks(self, kind: Kind) -> Iterable[Block]: ...
    def load_field(self, block: Block, count_hint: int) -> NDArray[float32]: ...
    def load(
        self, block: Block, count_hint: Optional[int] = None
    ) -> NDArray[float64]: ...
    @property
    def realizations(self) -> Set[int]: ...

def parse_name(name: str, kind: Kind) -> Tuple[str, int, int]: ...
