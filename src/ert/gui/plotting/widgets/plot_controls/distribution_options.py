import logging
from collections.abc import Callable

from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QGroupBox,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ert.gui.plotting.utils.qt_creator import create_group_box, create_group_layout

logger = logging.getLogger(__name__)


class DistributionOptions:
    def __init__(self, connection_point: Callable[..., object]) -> None:
        self._logged_options: set[str] = set()

        self._histogram = QCheckBox("Show histogram")
        self._histogram.setObjectName("histogram_checkbox")
        self._histogram.setToolTip(
            "Adds a histogram of the data to the plot."
            "\nDisplayed as counts or density depending on the selected option below."
        )
        self._histogram.setChecked(True)
        self._histogram.stateChanged.connect(connection_point)

        self._by_density = QRadioButton("Y axis by density")
        self._by_density.setObjectName("by_density_radiobutton")
        self._by_density.setToolTip(
            "Display histogram with density instead of counts."
            "\nThis may be useful for larger datasets."
        )

        self._by_count = QRadioButton("Y axis by count")
        self._by_count.setObjectName("by_count_radiobutton")
        self._by_count.setToolTip(
            "Display histogram with counts instead of density."
            "\nThis may be useful for smaller datasets."
        )

        self._histogram_options = QButtonGroup()
        self._histogram_options.addButton(self._by_density)
        self._histogram_options.addButton(self._by_count)
        self._by_density.setChecked(True)
        self._histogram_options.buttonClicked.connect(connection_point)

        # Container that can be indented and shown/hidden as a unit
        self._histogram_options_widget = QWidget()
        options_layout = QVBoxLayout(self._histogram_options_widget)
        options_layout.setContentsMargins(20, 0, 0, 0)
        options_layout.addWidget(self._by_density)
        options_layout.addWidget(self._by_count)

        # Toggle visibility with the histogram checkbox
        self._histogram.toggled.connect(self._histogram_options_widget.setVisible)
        self._histogram_options_widget.setVisible(self._histogram.isChecked())
        self._histogram.toggled.connect(self._reset_mode_when_histogram_off)

        self._gkde = QCheckBox("Show Gaussian KDE")
        self._gkde.setObjectName("gkde_checkbox")
        self._gkde.setToolTip(
            "Adds a Gaussian kernel density estimate to the plot."
            "\nDisplays a line for the probability density function"
            " of the data for each ensemble."
        )
        self._gkde.setChecked(True)
        self._gkde.stateChanged.connect(connection_point)

        self._rug_plot = QCheckBox("Show rug plot")
        self._rug_plot.setObjectName("rug_checkbox")
        self._rug_plot.setToolTip(
            "Displays the distribution as a rug plot for each ensemble."
            "\nIf histogram/Gaussian KDE is enabled, "
            "the rug plots will be plotted below the main plot."
        )
        self._rug_plot.setChecked(True)
        self._rug_plot.stateChanged.connect(connection_point)

        self._distribution_options = create_group_box(
            "Distribution options",
            create_group_layout(
                [
                    self._histogram,
                    self._histogram_options_widget,
                    self._gkde,
                    self._rug_plot,
                ]
            ),
        )

        self._histogram.clicked.connect(
            lambda checked: self._log_usage("Show histogram", checked)
        )
        self._by_density.clicked.connect(
            lambda checked: self._log_usage("Y axis by density", checked)
        )
        self._by_count.clicked.connect(
            lambda checked: self._log_usage("Y axis by count", checked)
        )
        self._gkde.clicked.connect(
            lambda checked: self._log_usage("Show Gaussian KDE", checked)
        )
        self._rug_plot.clicked.connect(
            lambda checked: self._log_usage("Show rug plot", checked)
        )

    @property
    def histogram_checkbox_state(self) -> bool:
        return self._histogram.isChecked()

    @histogram_checkbox_state.setter
    def histogram_checkbox_state(self, value: bool) -> None:
        self._histogram.setChecked(value)

    @property
    def gkde_checkbox_state(self) -> bool:
        return self._gkde.isChecked()

    @gkde_checkbox_state.setter
    def gkde_checkbox_state(self, value: bool) -> None:
        self._gkde.setChecked(value)

    @property
    def rug_checkbox_state(self) -> bool:
        return self._rug_plot.isChecked()

    @rug_checkbox_state.setter
    def rug_checkbox_state(self, value: bool) -> None:
        self._rug_plot.setChecked(value)

    def _reset_mode_when_histogram_off(self, checked: bool) -> None:
        if not checked:
            self._by_density.setChecked(True)

    def get_widget(self) -> QGroupBox:
        return self._distribution_options

    @property
    def histogram_by_density(self) -> bool:
        return self._by_density.isChecked()

    @histogram_by_density.setter
    def histogram_by_density(self, value: bool) -> None:
        self._by_density.setChecked(value)
        self._by_count.setChecked(not value)

    # Only wish to log the first time a distribution option is used in a session,
    # otherwise could risk flooding the log
    def _log_usage(self, distribution_option_name: str, _checked: bool) -> None:
        if distribution_option_name not in self._logged_options:
            logger.info("Plot sidebar option used: '%s'", distribution_option_name)
            self._logged_options.add(distribution_option_name)
