from __future__ import annotations

from enum import StrEnum, auto
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from pandas import DataFrame

    from ert.gui.tools.plot.plot_api import EnsembleObject

    from .plot_config import PlotConfig


class PlotType(StrEnum):
    LINE = auto()
    BAR = auto()
    SCATTER = auto()


class PlotContext:
    UNKNOWN_AXIS = None
    VALUE_AXIS = "VALUE"
    DATE_AXIS = "DATE"
    INDEX_AXIS = "INDEX"
    COUNT_AXIS = "COUNT"
    DENSITY_AXIS = "DENSITY"
    AXIS_TYPES: ClassVar[list[str | None]] = [
        UNKNOWN_AXIS,
        COUNT_AXIS,
        DATE_AXIS,
        DENSITY_AXIS,
        INDEX_AXIS,
        VALUE_AXIS,
    ]

    def __init__(
        self,
        plot_config: PlotConfig,
        ensembles: list[EnsembleObject],
        ensembles_color_indexes: list[int],
        key: str,
        layer: int | None = None,
    ) -> None:
        super().__init__()
        self._key = key
        self._ensembles = ensembles
        self._ensembles_color_indexes = ensembles_color_indexes
        self._plot_config = plot_config
        self.history_data: DataFrame | None = None
        self._layer: int | None = layer

        self._date_support_active = True
        self._x_axis: str | None = None
        self._y_axis: str | None = None
        self._log_scale = False
        self._by_batch: bool = True

        self._plot_type: PlotType | None = None

    @property
    def flip_response_axis(self) -> bool:
        return self._plot_config.flip_response_axis

    @flip_response_axis.setter
    def flip_response_axis(self, value: bool) -> None:
        self._plot_config.flip_response_axis = value

    @property
    def flip_observation_axis(self) -> bool:
        return self._plot_config.flip_observation_axis

    @flip_observation_axis.setter
    def flip_observation_axis(self, value: bool) -> None:
        self._plot_config.flip_observation_axis = value

    def plotConfig(self) -> PlotConfig:
        return self._plot_config

    def ensembles(self) -> list[EnsembleObject]:
        return self._ensembles

    def ensembles_color_indexes(self) -> list[int]:
        return self._ensembles_color_indexes

    def key(self) -> str:
        return self._key

    def deactivate_date_support(self) -> None:
        self._date_support_active = False

    def is_date_support_active(self) -> bool:
        return self._date_support_active

    @property
    def layer(self) -> int | None:
        return self._layer

    @property
    def x_axis(self) -> str | None:
        return self._x_axis

    @x_axis.setter
    def x_axis(self, value: str) -> None:
        if value not in PlotContext.AXIS_TYPES:
            raise UserWarning(
                f"Axis: '{value}' is not one of: {PlotContext.AXIS_TYPES}"
            )
        self._x_axis = value

    @property
    def y_axis(self) -> str | None:
        return self._y_axis

    @y_axis.setter
    def y_axis(self, value: str) -> None:
        if value not in PlotContext.AXIS_TYPES:
            raise UserWarning(
                f"Axis: '{value}' is not one of: {PlotContext.AXIS_TYPES}"
            )
        self._y_axis = value

    @property
    def plot_type(self) -> PlotType | None:
        return self._plot_type

    @plot_type.setter
    def plot_type(self, value: PlotType) -> None:
        self._plot_type = value

    def setXLabel(self, value: str) -> None:
        self._plot_config.set_x_label(value)

    def setYLabel(self, value: str) -> None:
        self._plot_config.set_y_label(value)

    @property
    def log_scale(self) -> bool:
        return self._log_scale

    @log_scale.setter
    def log_scale(self, value: bool) -> None:
        self._log_scale = value

    @property
    def by_batch(self) -> bool:
        return self._by_batch

    @by_batch.setter
    def by_batch(self, value: bool) -> None:
        self._by_batch = value
