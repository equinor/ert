from enum import StrEnum

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QSizePolicy, QWidget

from ert.gui.detect_mode import is_dark_mode

# EDS colour tokens
_PRIMARY_SELECTED_HIGHLIGHT = "#e6faec"
_PRIMARY_RESTING = "#007079"
_WARNING_HIGHLIGHT = "#ffe7d6"
_WARNING_TEXT = "#ad6200"


_MAX_TEXT_WIDTH = 240


class DataSource(StrEnum):
    CONFIG = "config"
    SELECTED_ENSEMBLE = "selected_ensemble"


# (text, background, text colour) for each source
_SOURCE_STYLES = {
    DataSource.CONFIG: (
        "From config file",
        _PRIMARY_SELECTED_HIGHLIGHT,
        _PRIMARY_RESTING,
    ),
    DataSource.SELECTED_ENSEMBLE: (
        "From selected ensemble",
        _WARNING_HIGHLIGHT,
        _WARNING_TEXT,
    ),
}


class SourceChip(QLabel):
    """EDS styled chip telling where the content of a summary card comes from.

    The config file and a selected ensemble use different colours so that it
    stands out when a card does not show what is in the config file.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("source_chip")
        self.setFixedHeight(22)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.source = DataSource.CONFIG
        self.set_source(DataSource.CONFIG)

    def set_source(self, source: DataSource, ensemble_name: str | None = None) -> None:
        """Show the source; for a selected ensemble, show its name if given."""
        self.source = source
        text, background, color = _SOURCE_STYLES[source]
        if source == DataSource.SELECTED_ENSEMBLE and ensemble_name:
            text = f"From {ensemble_name}"
        if is_dark_mode():
            background, color = color, "white"
        self.setStyleSheet(
            f"""
            QLabel#source_chip {{
                background-color: {background};
                color: {color};
                border-radius: 11px;
                padding: 0px 8px;
                font-size: 12px;
                font-weight: 500;
            }}
            """
        )
        # Keep long ensemble names from covering the card title; the full
        # text is in the tooltip.
        self.ensurePolished()
        elided = self.fontMetrics().elidedText(
            text, Qt.TextElideMode.ElideMiddle, _MAX_TEXT_WIDTH
        )
        self.setText(elided)
        self.setToolTip(text if elided != text else "")
        self.adjustSize()
