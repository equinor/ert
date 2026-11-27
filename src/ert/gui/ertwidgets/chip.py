from enum import StrEnum

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QSizePolicy, QWidget

_MAX_TEXT_WIDTH = 240


class DataSource(StrEnum):
    CONFIG = "config"
    SELECTED_ENSEMBLE = "selected_ensemble"


_SOURCE_TEXTS = {
    DataSource.CONFIG: "From config file",
    DataSource.SELECTED_ENSEMBLE: "From selected ensemble",
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
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.source = DataSource.CONFIG
        self.set_source(DataSource.CONFIG)

    def set_source(self, source: DataSource, ensemble_name: str | None = None) -> None:
        """Show the source; for a selected ensemble, show its name if given."""
        self.source = source
        text = _SOURCE_TEXTS[source]
        if source == DataSource.SELECTED_ENSEMBLE and ensemble_name:
            text = f"From {ensemble_name}"
        self.setProperty("source", source.value)
        style = self.style()
        assert style is not None
        style.unpolish(self)
        style.polish(self)
        # Keep long ensemble names from covering the card title; the full
        # text is in the tooltip.
        elided = self.fontMetrics().elidedText(
            text, Qt.TextElideMode.ElideMiddle, _MAX_TEXT_WIDTH
        )
        self.setText(elided)
        self.setToolTip(text if elided != text else "")
