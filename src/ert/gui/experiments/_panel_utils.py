from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget

from ert.gui.ertwidgets import (
    ActiveRealizationsModel,
    StringBox,
    TargetEnsembleModel,
    TextModel,
)
from ert.validation import ExperimentValidation, ProperNameFormatArgument
from ert.validation.active_range import ActiveRange
from ert.validation.range_string_argument import RangeSubsetStringArgument

if TYPE_CHECKING:
    from ert.config import AnalysisConfig
    from ert.gui.ertnotifier import ErtNotifier
    from ert.storage import Storage


def create_experiment_name_field(storage: Storage, mode: str) -> StringBox:
    name_field = StringBox(
        TextModel(""),
        placeholder_text=storage.get_unique_experiment_name(mode),
    )
    name_field.setMinimumWidth(250)
    name_field.setValidator(ExperimentValidation(storage))
    name_field.setObjectName("experiment_field")
    return name_field


def create_number_of_realizations_container(
    active_realizations_length: int,
) -> tuple[QWidget, QLabel]:
    number_of_realizations_container = QWidget()
    number_of_realizations_layout = QHBoxLayout(number_of_realizations_container)
    number_of_realizations_layout.setContentsMargins(0, 0, 0, 0)
    number_of_realizations_label = QLabel(f"<b>{active_realizations_length}</b>")
    number_of_realizations_label.setObjectName("num_reals_label")
    number_of_realizations_layout.addWidget(number_of_realizations_label)
    return number_of_realizations_container, number_of_realizations_label


def create_target_ensemble_format_field(
    analysis_config: AnalysisConfig, notifier: ErtNotifier
) -> tuple[TargetEnsembleModel, StringBox]:
    target_ensemble_format_model = TargetEnsembleModel(analysis_config, notifier)
    target_ensemble_format_field = StringBox(
        target_ensemble_format_model,  # type: ignore
        target_ensemble_format_model.getDefaultValue(),  # type: ignore
        continuous_update=True,
    )
    target_ensemble_format_field.setValidator(ProperNameFormatArgument())
    return target_ensemble_format_model, target_ensemble_format_field


def create_active_realizations_field(
    active_realizations: list[bool],
) -> tuple[StringBox, RangeSubsetStringArgument]:
    model = ActiveRealizationsModel(len(active_realizations))
    field = StringBox(
        model,  # type: ignore
        "config/experiment/active_realizations",
    )
    validator = RangeSubsetStringArgument(ActiveRange(active_realizations))
    field.setValidator(validator)
    model.setValueFromMask(active_realizations)
    return field, validator
