from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Self

import numpy as np

from ._read_summary import read_summary
from .parsing.config_dict import ConfigDict
from .parsing.config_errors import ConfigValidationError
from .parsing.config_keywords import ConfigKeys


@dataclass(eq=False)
class Refcase:
    start_date: datetime
    keys: list[str]
    dates: Sequence[datetime]
    values: list[list[float]]

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
    def all_dates(self) -> list[datetime]:
        return [self.start_date, *self.dates]

    @classmethod
    def from_config_dict(cls, config_dict: ConfigDict) -> Self | None:
        data = None
        refcase_file_path = config_dict.get(ConfigKeys.REFCASE)
        if refcase_file_path is not None:
            try:
                start_date, refcase_keys, time_map, data = read_summary(
                    str(refcase_file_path), ["*"]
                )
            except Exception as err:
                raise ConfigValidationError(f"Could not read refcase: {err}") from err

        return (
            cls(start_date, refcase_keys, time_map, data.tolist())
            if data is not None
            else None
        )
