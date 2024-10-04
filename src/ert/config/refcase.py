from dataclasses import dataclass
from datetime import datetime
from typing import (
    Any,
    List,
    Optional,
    Sequence,
)

import numpy as np
import numpy.typing as npt

from ._read_summary import read_summary
from .parsing.config_dict import ConfigDict
from .parsing.config_errors import ConfigValidationError
from .parsing.config_keywords import ConfigKeys


@dataclass(eq=False)
class Refcase:
    start_date: datetime
    keys: List[str]
    dates: Sequence[datetime]
    values: npt.NDArray[Any]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Refcase):
            return False
        return bool(
            self.start_date == other.start_date
            and self.keys == other.keys
            and self.dates == other.dates
            and np.all(self.values == other.values)
        )

    @property
    def all_dates(self) -> List[datetime]:
        return [self.start_date, *self.dates]

    @classmethod
    def from_config_dict(cls, config_dict: ConfigDict) -> Optional["Refcase"]:
        data = None
        refcase_file_path = config_dict.get(ConfigKeys.REFCASE)  # type: ignore
        if refcase_file_path is not None:
            try:
                start_date, refcase_keys, time_map, data = read_summary(
                    refcase_file_path, ["*"]
                )
            except Exception as err:
                raise ConfigValidationError(f"Could not read refcase: {err}") from err

        return (
            cls(start_date, refcase_keys, time_map, data) if data is not None else None
        )
