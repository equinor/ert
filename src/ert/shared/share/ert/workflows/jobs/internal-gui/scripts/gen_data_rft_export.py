import contextlib
import os
import re

import numpy
import pandas as pd
from qtpy.QtWidgets import QCheckBox

from ert.config import CancelPluginException, ErtPlugin


def load_args(filename, column_names=None):
    rows = 0
    columns = 0
    with open(filename, "r", encoding="utf-8") as fileH:
        for line in fileH.readlines():
            rows += 1
            columns = max(columns, len(line.split()))

    if column_names is not None:
        if len(column_names) <= columns:
            columns = len(column_names)
        else:
            raise ValueError("To many coloumns in input")

    data = numpy.empty(shape=(rows, columns), dtype=numpy.float64)
    data.fill(numpy.nan)

    row = 0
    with open(filename, encoding="utf-8") as fileH:
        for line in fileH.readlines():
            tmp = line.split()
            for column in range(columns):
                data[row][column] = float(tmp[column])
            row += 1

    if column_names is None:
        column_names = []
        for column in range(columns):
            column_names.append(f"Column{column:d}")

    data_frame = pd.DataFrame(data=data, columns=column_names)
    return data_frame


class GenDataRFTCSVExportJob(ErtPlugin):
    """Export of GEN_DATA based rfts to a CSV file. The csv file will in
     addition contain the depth as duplicated seperate row.

     The script expects four arguments:

       output_file: this is the path to the file to output the CSV data to

       key: this is the ert GEN_DATA key used for this particular RFT.

       report_step: This is the report step configured in the ert
         configuration file for this RFT.

       trajectory_file: This is the the file containing the

    Optional arguments:

     ensemble_list: a comma separated list of ensembles to export (no spaces allowed)
                if no list is provided the current ensemble is exported

     infer_iteration: If True the script will try to infer the iteration number
                by looking at the suffix of the ensemble name (i.e. default_2 = iteration 2)
                If False the script will use the ordering of the ensemble list: the first
                item will be iteration 0, the second item will be iteration 1...
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
        return "GEN_DATA RFT CSV Export"

    @staticmethod
    def getDescription():
        return "Export gen_data RFT results into a single CSV file."

    @staticmethod
    def inferIterationNumber(ensemble_name):
        pattern = re.compile("_([0-9]+$)")
        match = pattern.search(ensemble_name)

        if match is not None:
            return int(match.group(1))
        return 0

    def run(
        self,
        storage,
        workflow_args,
    ):
        """The run method will export the RFT's for all wells and all ensembles.

        The successful operation of this method hinges on two naming
        conventions:

          1. All the GEN_DATA RFT observations have key RFT_$WELL
          2. The trajectory files are in $trajectory_path/$WELL.txt
             or $trajectory_path/$WELL_R.txt

        """
        output_file = workflow_args[0]
        trajectory_path = workflow_args[1]
        ensemble_list = None if len(workflow_args) < 3 else workflow_args[2]
        _ = True if len(workflow_args) < 4 else workflow_args[3]
        drop_const_cols = False if len(workflow_args) < 5 else workflow_args[4]

        wells = set()

        ensemble_names = []
        if ensemble_list is not None:
            ensemble_names = ensemble_list.split(",")

        if len(ensemble_names) == 0:
            raise UserWarning("No ensembles given to load from")

        data = []
        for ensemble_name in ensemble_names:
            ensemble_name = ensemble_name.strip()
            ensemble_data = []

            try:
                ensemble = storage.get_ensemble_by_name(ensemble_name)
            except KeyError as exc:
                raise UserWarning(
                    f"The ensemble '{ensemble_name}' does not exist!"
                ) from exc

            if not ensemble.has_data():
                raise UserWarning(
                    f"The ensemble '{ensemble_name}' does not have any data!"
                )

            obs = ensemble.experiment.observations
            obs_keys = []
            for key, _ in obs.items():
                if key.startswith("RFT_"):
                    obs_keys.append(key)

            if len(obs_keys) == 0:
                raise UserWarning(
                    "The config does not contain any"
                    " GENERAL_OBSERVATIONS starting with RFT_*"
                )

            for obs_key in obs_keys:
                well = obs_key.replace("RFT_", "")
                wells.add(well)
                obs_vector = obs[obs_key]
                data_key = obs_vector.attrs["response"]
                if len(obs_vector.report_step) == 1:
                    report_step = obs_vector.report_step.values
                    obs_node = obs_vector.sel(report_step=report_step)
                else:
                    raise UserWarning(
                        "GEN_DATA RFT CSV Export can only be used for observations "
                        "active for exactly one report step"
                    )

                realizations = ensemble.get_realization_list_with_responses(data_key)
                vals = ensemble.load_responses(data_key, tuple(realizations)).sel(
                    report_step=report_step, drop=True
                )
                index = pd.Index(vals.index.values, name="axis")
                rft_data = pd.DataFrame(
                    data=vals["values"].values.reshape(len(vals.realization), -1).T,
                    index=index,
                    columns=realizations,
                )

                realizations = ensemble.get_realization_list_with_responses()

                # Trajectory
                trajectory_file = os.path.join(trajectory_path, f"{well}.txt")
                if not os.path.isfile(trajectory_file):
                    trajectory_file = os.path.join(trajectory_path, f"{well}_R.txt")

                arg = load_args(
                    trajectory_file, column_names=["utm_x", "utm_y", "md", "tvd"]
                )
                tvd_arg = arg["tvd"]

                # Observations
                for iens in realizations:
                    realization_frame = pd.DataFrame(
                        data={
                            "TVD": tvd_arg,
                            "Pressure": rft_data[iens],
                            "ObsValue": obs_node["observations"].values[0],
                            "ObsStd": obs_node["std"].values[0],
                        },
                        columns=["TVD", "Pressure", "ObsValue", "ObsStd"],
                    )

                    realization_frame["Realization"] = iens
                    realization_frame["Well"] = well
                    realization_frame["Ensemble"] = ensemble_name
                    realization_frame["Iteration"] = ensemble.iteration

                    ensemble_data.append(realization_frame)

                data.append(pd.concat(ensemble_data))

        frame = pd.concat(data)
        frame.set_index(["Realization", "Well", "Ensemble", "Iteration"], inplace=True)
        if drop_const_cols:
            frame = frame.loc[:, (frame != frame.iloc[0]).any()]

        frame.to_csv(output_file)
        well_list_str = ", ".join(list(wells))
        export_info = (
            f"Exported RFT information for wells: {well_list_str} to: {output_file}"
        )
        return export_info

    def getArguments(self, parent, storage):
        from ert.gui.ertwidgets import CustomDialog, ListEditBox, PathChooser
        from ert.gui.ertwidgets.models.path_model import PathModel

        description = (
            "The GEN_DATA RFT CSV export requires some information before it starts:"
        )
        dialog = CustomDialog("Robust CSV Export", description, parent)

        output_path_model = PathModel("output.csv")
        output_path_chooser = PathChooser(output_path_model)

        trajectory_model = PathModel(
            "wellpath", must_be_a_directory=True, must_be_a_file=False, must_exist=True
        )
        trajectory_chooser = PathChooser(trajectory_model)
        trajectory_chooser.setObjectName("trajectory_chooser")

        all_ensemble_list = [ensemble.name for ensemble in storage.ensembles]
        list_edit = ListEditBox(all_ensemble_list)
        list_edit.setObjectName("list_of_ensembles")

        infer_iteration_check = QCheckBox()
        infer_iteration_check.setChecked(True)
        infer_iteration_check.setToolTip(GenDataRFTCSVExportJob.INFER_HELP)

        drop_const_columns_check = QCheckBox()
        drop_const_columns_check.setChecked(False)
        drop_const_columns_check.setToolTip(
            "If checked, exclude columns whose value is the same for every entry"
        )

        dialog.addLabeledOption("Output file path", output_path_chooser)
        dialog.addLabeledOption("Trajectory file", trajectory_chooser)
        dialog.addLabeledOption("List of ensembles to export", list_edit)
        dialog.addLabeledOption("Infer iteration number", infer_iteration_check)
        dialog.addLabeledOption("Drop constant columns", drop_const_columns_check)

        dialog.addButtons()

        success = dialog.showAndTell()

        if success:
            ensemble_list = ",".join(list_edit.getItems())
            with contextlib.suppress(ValueError):
                return [
                    output_path_model.getPath(),
                    trajectory_model.getPath(),
                    ensemble_list,
                    infer_iteration_check.isChecked(),
                    drop_const_columns_check.isChecked(),
                ]

        raise CancelPluginException("User cancelled!")
