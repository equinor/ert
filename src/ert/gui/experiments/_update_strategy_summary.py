from collections import Counter
from collections.abc import Iterable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QLabel,
    QWidget,
)

from ert.config.parameter_config import LocalizationType, ParameterConfig

_PARAMETER_TYPE_DISPLAY_NAMES = {
    "gen_kw": "GenKW",
    "field": "Field",
    "surface": "Surface",
}


class UpdateStrategySummary(QLabel):
    def __init__(
        self,
        parameter_configs: Iterable[ParameterConfig],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("update_strategy_summary")
        self.setAccessibleName("Update strategy counts by parameter type")
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setWordWrap(True)
        self.setToolTip(
            "Each Field or Surface counts as one configuration, "
            "regardless of its number of grid cells."
        )
        self.set_parameters(parameter_configs)

    def set_parameters(self, parameter_configs: Iterable[ParameterConfig]) -> None:
        counts = Counter((p.update_strategy, p.type) for p in parameter_configs)
        present_strategies = {strategy for strategy, _ in counts}
        strategies = [
            strategy
            for strategy in (*LocalizationType, None)
            if strategy in present_strategies
        ]

        strategy_summaries = []
        for strategy in strategies:
            parameter_types = sorted(
                param_type
                for counted_strategy, param_type in counts
                if counted_strategy == strategy
            )
            parameter_summaries = [
                (
                    f"{counts[strategy, param_type]:,} "
                    f"{_PARAMETER_TYPE_DISPLAY_NAMES.get(param_type, param_type)}"
                )
                for param_type in parameter_types
            ]
            strategy_name = (
                strategy.value.capitalize() if strategy is not None else "Non-updatable"
            )
            strategy_summaries.append(
                f"{strategy_name}: {', '.join(parameter_summaries)}"
            )

        self.setText("; ".join(strategy_summaries) if strategy_summaries else "None")
