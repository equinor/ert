from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from ert.gui.plottery import PlotConfig


class PlotContext:
    UNKNOWN_AXIS = None
    VALUE_AXIS = "VALUE"
    DATE_AXIS = "DATE"
    INDEX_AXIS = "INDEX"
    COUNT_AXIS = "COUNT"
    DENSITY_AXIS = "DENSITY"
    DEPTH_AXIS = "DEPTH"
    AXIS_TYPES = [
        UNKNOWN_AXIS,
        COUNT_AXIS,
        DATE_AXIS,
        DENSITY_AXIS,
        DEPTH_AXIS,
        INDEX_AXIS,
        VALUE_AXIS,
    ]

    def __init__(self, plot_config, cases, key):
        super().__init__()
        self._key = key
        self._cases = cases
        self._plot_config = plot_config
        self.history_data = None
        self._log_scale = False

        self._date_support_active = True
        self._x_axis = None
        self._y_axis = None

    def plotConfig(self) -> "PlotConfig":
        return self._plot_config

    def cases(self) -> List[str]:
        return self._cases

    def key(self) -> str:
        return self._key

    def deactivateDateSupport(self):
        self._date_support_active = False

    def isDateSupportActive(self) -> bool:
        return self._date_support_active

    @property
    def x_axis(self) -> str:
        return self._x_axis

    @x_axis.setter
    def x_axis(self, value: str):
        if value not in PlotContext.AXIS_TYPES:
            raise UserWarning(
                f"Axis: '{value}' is not one of: {PlotContext.AXIS_TYPES}"
            )
        self._x_axis = value

    @property
    def y_axis(self) -> str:
        return self._y_axis

    @y_axis.setter
    def y_axis(self, value: str):
        if value not in PlotContext.AXIS_TYPES:
            raise UserWarning(
                f"Axis: '{value}' is not one of: {PlotContext.AXIS_TYPES}"
            )
        self._y_axis = value

    @property
    def log_scale(self):
        return self._log_scale

    @log_scale.setter
    def log_scale(self, value):
        self._log_scale = value
