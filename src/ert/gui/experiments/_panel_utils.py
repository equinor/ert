from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QFormLayout, QHBoxLayout, QLabel, QWidget

from ert.config.design_matrix import DesignMatrix
from ert.gui.ertwidgets import (
    ActiveRealizationsModel,
    StringBox,
    TargetEnsembleModel,
    TextModel,
    get_parameters_button,
)
from ert.validation import ExperimentValidation, ProperNameFormatArgument
from ert.validation.active_range import ActiveRange
from ert.validation.range_string_argument import RangeSubsetStringArgument

from ._design_matrix_panel import DesignMatrixPanel

if TYPE_CHECKING:
    from collections.abc import Callable

    from ert.config import AnalysisConfig, ParameterConfig
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


def merge_design_matrix_parameters(
    design_matrix: DesignMatrix | None,
    parameter_configuration: list[ParameterConfig],
) -> list[ParameterConfig]:
    if design_matrix is None:
        return parameter_configuration

    return design_matrix.merge_with_existing_parameters(parameter_configuration)


def add_parameter_configuration_rows(
    parent: QWidget,
    layout: QFormLayout,
    design_matrix: DesignMatrix | None,
    get_parameter_configuration: Callable[[], list[ParameterConfig]],
    *,
    number_of_realizations_label: QLabel | None = None,
    config_num_realization: int | None = None,
) -> None:
    if design_matrix is not None:
        layout.addRow(
            "Design matrix",
            DesignMatrixPanel.get_design_matrix_button(
                design_matrix,
                number_of_realizations_label,
                config_num_realization,
            ),
        )

    if get_parameter_configuration():
        layout.addRow(
            "Parameters", get_parameters_button(get_parameter_configuration, parent)
        )
