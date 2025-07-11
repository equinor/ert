import contextlib
import json
import logging
import os
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from PyQt6.QtWidgets import QCheckBox, QWidget

from ert.plugins import CancelPluginException, ErtPlugin
from ert.storage import Storage

logger = logging.getLogger(__name__)


def load_args(filename: str, column_names: list[str] | None = None) -> pd.DataFrame:
    rows = 0
    columns = 0
    with open(filename, encoding="utf-8") as fileH:
        for line in fileH:
            rows += 1
            columns = max(columns, len(line.split()))

    if column_names is not None:
        if len(column_names) <= columns:
            columns = len(column_names)
        else:
            raise ValueError("To many columns in input")

    data = np.empty(shape=(rows, columns), dtype=np.float64)
    data.fill(np.nan)

    row = 0
    with open(filename, encoding="utf-8") as fileH:
        for line in fileH:
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

       ensemble_data_as_json: a comma separated list of ensembles to
           export (no spaces allowed). If no list is provided the current
           ensemble is exported

    """

    @staticmethod
    def getName() -> str:
        return "GEN_DATA RFT CSV Export"

    @staticmethod
    def getDescription() -> str:
        return "Export gen_data RFT results into a single CSV file."

    def run(
        self,
        storage: Storage,
        workflow_args: Sequence[str],
    ) -> str:
        """The run method will export the RFT's for all wells and all ensembles."""

        output_file = workflow_args[0]
        trajectory_path = workflow_args[1]
        ensemble_data_as_json = None if len(workflow_args) < 3 else workflow_args[2]
        drop_const_cols = False if len(workflow_args) < 4 else bool(workflow_args[3])

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

            obs_df = ensemble.experiment.observations.get("gen_data")
            obs_keys = []
            for key in ensemble.experiment.observation_keys:
                if key.startswith("RFT_"):
                    obs_keys.append(key)

            if len(obs_keys) == 0 or obs_df is None:
                raise UserWarning(
                    "The config does not contain any"
                    " GENERAL_OBSERVATIONS starting with RFT_*"
                )

            for obs_key in obs_keys:
                well_key = obs_key.replace("RFT_", "")

                obs_df = obs_df.filter(pl.col("observation_key").eq(obs_key))
                response_key = obs_df["response_key"].unique().to_list()[0]

                if len(obs_df["report_step"].unique()) != 1:
                    raise UserWarning(
                        "GEN_DATA RFT CSV Export can only be used for observations "
                        "active for exactly one report step"
                    )

                realizations = ensemble.get_realization_list_with_responses()
                responses = ensemble.load_responses(response_key, tuple(realizations))
                joined = obs_df.join(
                    responses,
                    on=["response_key", "report_step", "index"],
                    how="left",
                ).drop("index", "report_step")

                # Trajectory
                trajectory_file = os.path.join(trajectory_path, f"{well_key}.txt")
                if not os.path.isfile(trajectory_file):
                    trajectory_file = os.path.join(trajectory_path, f"{well_key}_R.txt")

                arg = load_args(
                    trajectory_file, column_names=["utm_x", "utm_y", "md", "tvd"]
                )
                tvd_arg = arg["tvd"]

                all_realization_frames = joined.rename(
                    {
                        "realization": "Realization",
                        "values": "Pressure",
                        "observations": "ObsValue",
                        "std": "ObsStd",
                    }
                ).with_columns(
                    [
                        pl.lit(well_key).alias("Well").cast(pl.String),
                        pl.lit(ensemble.name).alias("Ensemble").cast(pl.String),
                        pl.lit(ensemble.iteration).alias("Iteration").cast(pl.UInt8),
                        pl.lit(tvd_arg).alias("TVD").cast(pl.Float32),
                    ]
                )

                data.append(all_realization_frames)

        frame = pl.concat(data)

        cols_index = ["Well", "Ensemble", "Iteration"]
        const_cols_right = ["ObsValue", "ObsStd"]
        const_cols_left = [
            col
            for col in frame.columns
            if (
                col not in cols_index
                and col not in const_cols_right
                and frame[col].n_unique() == 1
            )
        ]

        columns_to_export = [
            "Realization",
            *cols_index,
            *(const_cols_left if not drop_const_cols else []),
            *["Pressure"],
            *(const_cols_right if not drop_const_cols else []),
        ]

        to_export = frame.select(columns_to_export)

        to_export.write_csv(output_file, include_header=True)
        well_list_str = ", ".join(to_export["Well"].unique().to_list())
        export_info = (
            f"Exported RFT information for wells: {well_list_str} to: {output_file}"
        )
        return export_info

    def getArguments(self, parent: QWidget, storage: Storage) -> list[Any]:  # type: ignore
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
            logger.info("Gui utility: Gen Data RFT CSV export was used")
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
