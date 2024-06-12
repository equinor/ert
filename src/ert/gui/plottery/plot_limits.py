from dataclasses import dataclass
from datetime import date
from typing import Optional, Tuple, Union

Num = Union[float, int]


@dataclass
class PlotLimits:
    value_limits: Tuple[Optional[Num], Optional[Num]] = (None, None)
    index_limits: Tuple[Optional[int], Optional[int]] = (None, None)
    count_limits: Tuple[Optional[int], Optional[int]] = (None, None)
    density_limits: Tuple[Optional[Num], Optional[Num]] = (None, None)
    date_limits: Tuple[Optional[date], Optional[date]] = (None, None)

    @property
    def value_minimum(self) -> Optional[Num]:
        return self.value_limits[0]

    @value_minimum.setter
    def value_minimum(self, value: Optional[Num]) -> None:
        self.value_limits = (value, self.value_limits[1])

    @property
    def value_maximum(self) -> Optional[Num]:
        return self.value_limits[1]

    @value_maximum.setter
    def value_maximum(self, value: Optional[Num]) -> None:
        self.value_limits = (self.value_limits[0], value)

    @property
    def count_minimum(self) -> Optional[int]:
        return self.count_limits[0]

    @count_minimum.setter
    def count_minimum(self, value: Optional[int]) -> None:
        self.count_limits = (value, self.count_limits[1])

    @property
    def count_maximum(self) -> Optional[int]:
        return self.count_limits[1]

    @count_maximum.setter
    def count_maximum(self, value: Optional[int]) -> None:
        self.count_limits = (self.count_limits[0], value)

    @property
    def index_minimum(self) -> Optional[int]:
        return self.index_limits[0]

    @index_minimum.setter
    def index_minimum(self, value: Optional[int]) -> None:
        self.index_limits = (value, self.index_limits[1])

    @property
    def index_maximum(self) -> Optional[int]:
        return self.index_limits[1]

    @index_maximum.setter
    def index_maximum(self, value: Optional[int]) -> None:
        self.index_limits = (self.index_limits[0], value)

    @property
    def density_minimum(self) -> Optional[Num]:
        return self.density_limits[0]

    @density_minimum.setter
    def density_minimum(self, value: Optional[Num]) -> None:
        self.density_limits = (value, self.density_limits[1])

    @property
    def density_maximum(self) -> Optional[Num]:
        return self.density_limits[1]

    @density_maximum.setter
    def density_maximum(self, value: Optional[Num]) -> None:
        self.density_limits = (self.density_limits[0], value)

    @property
    def date_minimum(self) -> Optional[date]:
        return self.date_limits[0]

    @date_minimum.setter
    def date_minimum(self, value: Optional[date]) -> None:
        self.date_limits = (value, self.date_limits[1])

    @property
    def date_maximum(self) -> Optional[date]:
        return self.date_limits[1]

    @date_maximum.setter
    def date_maximum(self, value: Optional[date]) -> None:
        self.date_limits = (self.date_limits[0], value)
