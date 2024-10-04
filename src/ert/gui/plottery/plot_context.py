from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, List, Optional

from ert.gui.tools.plot.plot_api import EnsembleObject

if TYPE_CHECKING:
    from pandas import DataFrame

    from ert.gui.plottery import PlotConfig


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
        ensembles: List[EnsembleObject],
        key: str,
        layer: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._key = key
        self._ensembles = ensembles
        self._plot_config = plot_config
        self.history_data: Optional[DataFrame] = None
        self._log_scale = False
        self._layer: Optional[int] = layer

        self._date_support_active = True
        self._x_axis: Optional[str] = None
        self._y_axis: Optional[str] = None

    def plotConfig(self) -> "PlotConfig":
        return self._plot_config

    def ensembles(self) -> List[EnsembleObject]:
        return self._ensembles

    def key(self) -> str:
        return self._key

    def deactivateDateSupport(self) -> None:
        self._date_support_active = False

    def isDateSupportActive(self) -> bool:
        return self._date_support_active

    @property
    def layer(self) -> Optional[int]:
        return self._layer

    @property
    def x_axis(self) -> Optional[str]:
        return self._x_axis

    @x_axis.setter
    def x_axis(self, value: str) -> None:
        if value not in PlotContext.AXIS_TYPES:
            raise UserWarning(
                f"Axis: '{value}' is not one of: {PlotContext.AXIS_TYPES}"
            )
        self._x_axis = value

    @property
    def y_axis(self) -> Optional[str]:
        return self._y_axis

    @y_axis.setter
    def y_axis(self, value: str) -> None:
        if value not in PlotContext.AXIS_TYPES:
            raise UserWarning(
                f"Axis: '{value}' is not one of: {PlotContext.AXIS_TYPES}"
            )
        self._y_axis = value

    @property
    def log_scale(self) -> bool:
        return self._log_scale

    @log_scale.setter
    def log_scale(self, value: bool) -> None:
        self._log_scale = value
