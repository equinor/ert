from dataclasses import dataclass
from datetime import date
from typing import TypeAlias

Num: TypeAlias = float | int


@dataclass
class PlotLimits:
    value_limits: tuple[Num | None, Num | None] = (None, None)
    index_limits: tuple[int | None, int | None] = (None, None)
    count_limits: tuple[int | None, int | None] = (None, None)
    density_limits: tuple[Num | None, Num | None] = (None, None)
    date_limits: tuple[date | None, date | None] = (None, None)

    @property
    def value_minimum(self) -> Num | None:
        return self.value_limits[0]

    @value_minimum.setter
    def value_minimum(self, value: Num | None) -> None:
        self.value_limits = (value, self.value_limits[1])

    @property
    def value_maximum(self) -> Num | None:
        return self.value_limits[1]

    @value_maximum.setter
    def value_maximum(self, value: Num | None) -> None:
        self.value_limits = (self.value_limits[0], value)

    @property
    def count_minimum(self) -> int | None:
        return self.count_limits[0]

    @count_minimum.setter
    def count_minimum(self, value: int | None) -> None:
        self.count_limits = (value, self.count_limits[1])

    @property
    def count_maximum(self) -> int | None:
        return self.count_limits[1]

    @count_maximum.setter
    def count_maximum(self, value: int | None) -> None:
        self.count_limits = (self.count_limits[0], value)

    @property
    def index_minimum(self) -> int | None:
        return self.index_limits[0]

    @index_minimum.setter
    def index_minimum(self, value: int | None) -> None:
        self.index_limits = (value, self.index_limits[1])

    @property
    def index_maximum(self) -> int | None:
        return self.index_limits[1]

    @index_maximum.setter
    def index_maximum(self, value: int | None) -> None:
        self.index_limits = (self.index_limits[0], value)

    @property
    def density_minimum(self) -> Num | None:
        return self.density_limits[0]

    @density_minimum.setter
    def density_minimum(self, value: Num | None) -> None:
        self.density_limits = (value, self.density_limits[1])

    @property
    def density_maximum(self) -> Num | None:
        return self.density_limits[1]

    @density_maximum.setter
    def density_maximum(self, value: Num | None) -> None:
        self.density_limits = (self.density_limits[0], value)

    @property
    def date_minimum(self) -> date | None:
        return self.date_limits[0]

    @date_minimum.setter
    def date_minimum(self, value: date | None) -> None:
        self.date_limits = (value, self.date_limits[1])

    @property
    def date_maximum(self) -> date | None:
        return self.date_limits[1]

    @date_maximum.setter
    def date_maximum(self, value: date | None) -> None:
        self.date_limits = (self.date_limits[0], value)
