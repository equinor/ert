from __future__ import annotations

import json
from typing import TYPE_CHECKING, Optional

from qtpy.QtWidgets import QCheckBox, QWidget

from ert.gui.ertwidgets import (
    CustomDialog,
    ListEditBox,
    PathChooser,
    PathModel,
)

if TYPE_CHECKING:
    from ert.config import ErtConfig
    from ert.storage import LocalStorage


class ExportDialog(CustomDialog):
    def __init__(
        self,
        ert_config: ErtConfig,
        storage: LocalStorage,
        parent: Optional[QWidget] = None,
    ) -> None:
        description = "The CSV export requires some information before it starts:"
        super().__init__("export", description, parent)
        self.storage = storage  # Store the storage reference
        subs_list = ert_config.substitution_list
        default_csv_output_path = subs_list.get("<CSV_OUTPUT_PATH>", "output.csv")
        self.output_path_model = PathModel(default_csv_output_path)
        output_path_chooser = PathChooser(self.output_path_model)
        design_matrix_default = subs_list.get("<DESIGN_MATRIX_PATH>", "")
        self.design_matrix_path_model = PathModel(
            design_matrix_default, is_required=False, must_exist=True
        )
        design_matrix_path_chooser = PathChooser(self.design_matrix_path_model)

        # Create a dictionary of ensemble names to their IDs and experiment names
        self.ensemble_dict = {
            ensemble.name: {
                "ensemble_id": str(ensemble.id),
                "experiment_name": ensemble.experiment.name,
            }
            for ensemble in storage.ensembles
            if ensemble.has_data()
        }

        self.list_edit = ListEditBox(list(self.ensemble_dict.keys()))

        self.drop_const_columns_check = QCheckBox()
        self.drop_const_columns_check.setChecked(False)
        self.drop_const_columns_check.setToolTip(
            "If checked, exclude columns whose value is the same for every entry"
        )
        self.addLabeledOption("Output file path", output_path_chooser)
        self.addLabeledOption("Design matrix path", design_matrix_path_chooser)
        self.addLabeledOption("List of ensembles to export", self.list_edit)
        self.addLabeledOption("Drop constant columns", self.drop_const_columns_check)
        self.addButtons()

    @property
    def output_path(self) -> Optional[str]:
        return self.output_path_model.getPath()

    @property
    def ensemble_list(self) -> str:
        selected_ensembles = self.list_edit.getItems()
        selected_dict = {
            self.ensemble_dict[name]["ensemble_id"]: {
                "name": name,
                "experiment_name": self.ensemble_dict[name]["experiment_name"],
            }
            for name in selected_ensembles
            if name in self.ensemble_dict
        }
        return json.dumps(selected_dict)

    @property
    def design_matrix_path(self) -> Optional[str]:
        path = self.design_matrix_path_model.getPath()
        if not path or not path.strip():
            path = None
        return path

    @property
    def drop_const_columns(self) -> bool:
        return self.drop_const_columns_check.isChecked()
