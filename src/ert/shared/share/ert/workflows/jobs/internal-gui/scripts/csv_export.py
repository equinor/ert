import os
import re

import pandas
from qtpy.QtWidgets import QCheckBox

from ert import LibresFacade
from ert.config import CancelPluginException, ErtPlugin


def loadDesignMatrix(filename) -> pandas.DataFrame:
    dm = pandas.read_csv(filename, delim_whitespace=True)
    dm = dm.rename(columns={dm.columns[0]: "Realization"})
    dm = dm.set_index(["Realization"])
    return dm


class CSVExportJob(ErtPlugin):
    """
    Export of summary, misfit, design matrix data and gen kw into a single CSV file.

    The script expects a single argument:

    output_file: this is the path to the file to output the CSV data to

    Optional arguments:

    ensemble_list: a comma separated list of ensembles to export (no spaces allowed)
               if no list is provided the current ensemble is exported
               a single * can be used to export all ensembles

    design_matrix: a path to a file containing the design matrix

    infer_iteration: If True the script will try to infer the iteration number
                     by looking at the suffix of the ensemble name (i.e. default_2
                     = iteration 2). If False the script will use the ordering
                     of the ensemble list: the first item will be iteration 0,
                     the second item will be iteration 1...

    The script also looks for default values for output path and design matrix
    path to present in the GUI. These can be specified with DATA_KW keyword in
    the config file:
        DATA_KW <CSV_OUTPUT_PATH> {some path}
        DATA_KW <DESIGN_MATRIX_PATH> {some path}
    """

    INFER_HELP = (
        "<html>"
        "If this is checked the iteration number will be inferred from the name i.e.:"
        "<ul>"
        "<li>ensemble_name -> iteration: 0</li>"
        "<li>ensemble_name_0 -> iteration: 0</li>"
        "<li>ensemble_name_2 -> iteration: 2</li>"
        "<li>ensemble_0, ensemble_2, ensemble_5 -> iterations: 0, 2, 5</li>"
        "</ul>"
        "Leave this unchecked to set iteration number to the order of the listed ensembles:"
        "<ul><li>ensemble_0, ensemble_2, ensemble_5 -> iterations: 0, 1, 2</li></ul>"
        "<br/>"
        "</html>"
    )

    @staticmethod
    def getName():
        return "CSV Export"

    @staticmethod
    def getDescription():
        return (
            "Export GenKW, design matrix, misfit data "
            "and summary data into a single CSV file."
        )

    @staticmethod
    def inferIterationNumber(ensemble_name):
        pattern = re.compile("_([0-9]+$)")
        match = pattern.search(ensemble_name)

        if match is not None:
            return int(match.group(1))
        return 0

    def run(
        self,
        ert_config,
        storage,
        workflow_args,
    ):
        output_file = workflow_args[0]
        ensemble_list = None if len(workflow_args) < 2 else workflow_args[1]
        design_matrix_path = None if len(workflow_args) < 3 else workflow_args[2]
        _ = True if len(workflow_args) < 4 else workflow_args[3]
        drop_const_cols = False if len(workflow_args) < 5 else workflow_args[4]
        ensembles = []
        facade = LibresFacade(ert_config)

        if ensemble_list is not None:
            if ensemble_list.strip() == "*":
                ensembles = self.getAllEnsembleList(storage)
            else:
                ensembles = ensemble_list.split(",")

        if ensemble_list is None or len(ensembles) == 0:
            ensembles = "default"

        if design_matrix_path is not None:
            if not os.path.exists(design_matrix_path):
                raise UserWarning("The design matrix file does not exists!")

            if not os.path.isfile(design_matrix_path):
                raise UserWarning("The design matrix is not a file!")

        data = pandas.DataFrame()

        for ensemble in ensembles:
            ensemble = ensemble.strip()

            try:
                ensemble = self.storage.get_ensemble_by_name(ensemble)
            except KeyError as exc:
                raise UserWarning(f"The ensemble '{ensemble}' does not exist!") from exc

            if not ensemble.has_data():
                raise UserWarning(f"The ensemble '{ensemble}' does not have any data!")

            ensemble_data = ensemble.load_all_gen_kw_data()

            if design_matrix_path is not None:
                design_matrix_data = loadDesignMatrix(design_matrix_path)
                if not design_matrix_data.empty:
                    ensemble_data = ensemble_data.join(design_matrix_data, how="outer")

            misfit_data = facade.load_all_misfit_data(ensemble)
            if not misfit_data.empty:
                ensemble_data = ensemble_data.join(misfit_data, how="outer")

            summary_data = ensemble.load_all_summary_data()
            if not summary_data.empty:
                ensemble_data = ensemble_data.join(summary_data, how="outer")
            else:
                ensemble_data["Date"] = None
                ensemble_data.set_index(["Date"], append=True, inplace=True)

            ensemble_data["Iteration"] = ensemble.iteration
            ensemble_data["Ensemble"] = ensemble.name
            ensemble_data.set_index(
                ["Ensemble", "Iteration"], append=True, inplace=True
            )

            data = pandas.concat([data, ensemble_data])

        data = data.reorder_levels(["Realization", "Iteration", "Date", "Ensemble"])
        if drop_const_cols:
            data = data.loc[:, (data != data.iloc[0]).any()]

        data.to_csv(output_file)

        export_info = (
            f"Exported {len(data.index)} rows and {len(data.columns)} "
            f"columns to {output_file}."
        )
        return export_info

    def getArguments(self, parent, ert_config, storage):
        from ert.gui.ertwidgets.customdialog import CustomDialog
        from ert.gui.ertwidgets.listeditbox import ListEditBox
        from ert.gui.ertwidgets.models.path_model import PathModel
        from ert.gui.ertwidgets.pathchooser import PathChooser

        description = "The CSV export requires some information before it starts:"
        dialog = CustomDialog("CSV Export", description, parent)

        subs_list = ert_config.substitution_list
        default_csv_output_path = subs_list.get("<CSV_OUTPUT_PATH>", "output.csv")
        output_path_model = PathModel(default_csv_output_path)
        output_path_chooser = PathChooser(output_path_model)

        design_matrix_default = subs_list.get("<DESIGN_MATRIX_PATH>", "")
        design_matrix_path_model = PathModel(
            design_matrix_default, is_required=False, must_exist=True
        )
        design_matrix_path_chooser = PathChooser(design_matrix_path_model)

        list_edit = ListEditBox(self.getAllEnsembleList(storage))

        infer_iteration_check = QCheckBox()
        infer_iteration_check.setChecked(True)
        infer_iteration_check.setToolTip(CSVExportJob.INFER_HELP)

        drop_const_columns_check = QCheckBox()
        drop_const_columns_check.setChecked(False)
        drop_const_columns_check.setToolTip(
            "If checked, exclude columns whose value is the same for every entry"
        )

        dialog.addLabeledOption("Output file path", output_path_chooser)
        dialog.addLabeledOption("Design matrix path", design_matrix_path_chooser)
        dialog.addLabeledOption("List of ensembles to export", list_edit)
        dialog.addLabeledOption("Infer iteration number", infer_iteration_check)
        dialog.addLabeledOption("Drop constant columns", drop_const_columns_check)

        dialog.addButtons()

        success = dialog.showAndTell()

        if success:
            design_matrix_path = design_matrix_path_model.getPath()
            if not design_matrix_path.strip():
                design_matrix_path = None

            ensemble_list = ",".join(list_edit.getItems())

            return [
                output_path_model.getPath(),
                ensemble_list,
                design_matrix_path,
                infer_iteration_check.isChecked(),
                drop_const_columns_check.isChecked(),
            ]

        raise CancelPluginException("User cancelled!")

    @staticmethod
    def getAllEnsembleList(storage):
        all_ensemble_list = [
            ensemble.name for ensemble in storage.ensembles if ensemble.has_data()
        ]
        return all_ensemble_list
