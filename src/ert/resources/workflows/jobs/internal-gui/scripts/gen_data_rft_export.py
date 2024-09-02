import contextlib
import json
import os

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

     ensemble_data_as_json: a comma separated list of ensembles to export (no spaces allowed)
                if no list is provided the current ensemble is exported

    """

    @staticmethod
    def getName():
        return "GEN_DATA RFT CSV Export"

    @staticmethod
    def getDescription():
        return "Export gen_data RFT results into a single CSV file."

    def run(
        self,
        storage,
        workflow_args,
    ):
        """The run method will export the RFT's for all wells and all ensembles."""

        output_file = workflow_args[0]
        trajectory_path = workflow_args[1]
        ensemble_data_as_json = None if len(workflow_args) < 3 else workflow_args[2]
        drop_const_cols = False if len(workflow_args) < 4 else bool(workflow_args[3])

        wells = set()

        ensemble_data_as_dict = (
            json.loads(ensemble_data_as_json) if ensemble_data_as_json else {}
        )

        if not ensemble_data_as_dict:
            raise UserWarning("No ensembles given to load from")

        data = []
        for ensemble_id, ensemble_info in ensemble_data_as_dict.items():
            ensemble_name = ensemble_info["ensemble_name"]

            try:
                ensemble = storage.get_ensemble(ensemble_id)
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

            ensemble_data = []
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
        # Importing ert.gui on-demand saves ~0.5 seconds off `from ert import __main__`
        from ert.gui.ertwidgets import (  # noqa: PLC0415
            CustomDialog,
            ListEditBox,
            PathChooser,
        )
        from ert.gui.ertwidgets.models.path_model import PathModel  # noqa: PLC0415

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

        ensemble_with_data_dict = {
            ensemble.id: ensemble.name
            for ensemble in storage.ensembles
            if ensemble.has_data()
        }
        list_edit = ListEditBox(ensemble_with_data_dict)
        list_edit.setObjectName("list_of_ensembles")

        drop_const_columns_check = QCheckBox()
        drop_const_columns_check.setChecked(False)
        drop_const_columns_check.setObjectName("drop_const_columns_check")
        drop_const_columns_check.setToolTip(
            "If checked, exclude columns whose value is the same for every entry"
        )

        dialog.addLabeledOption("Output file path", output_path_chooser)
        dialog.addLabeledOption("Trajectory file", trajectory_chooser)
        dialog.addLabeledOption("List of ensembles to export", list_edit)
        dialog.addLabeledOption("Drop constant columns", drop_const_columns_check)

        dialog.addButtons()

        success = dialog.showAndTell()

        if success:
            ensemble_data_as_dict = {
                str(ensemble.id): {
                    "ensemble_name": ensemble.name,
                    "experiment_name": ensemble.experiment.name,
                }
                for ensemble in storage.ensembles
                if ensemble.name in list_edit.getItems().values()
            }
            with contextlib.suppress(ValueError):
                return [
                    output_path_model.getPath(),
                    trajectory_model.getPath(),
                    json.dumps(
                        ensemble_data_as_dict
                    ),  # Return the ensemble list as a JSON string
                    drop_const_columns_check.isChecked(),
                ]

        raise CancelPluginException("User cancelled!")
