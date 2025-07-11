import json
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import pandas as pd
import polars as pl

from ert import ErtScript, LibresFacade
from ert.storage import Storage

if TYPE_CHECKING:
    from ert.storage import Ensemble


def loadDesignMatrix(filename: str) -> pd.DataFrame:
    dm = pd.read_csv(filename, delim_whitespace=True)
    dm = dm.rename(columns={dm.columns[0]: "Realization"})
    dm = dm.set_index(["Realization"])
    return dm


class CSVExportJob(ErtScript):
    """Export of summary, misfit, design matrix data and gen kw into a single CSV file.

    The script expects a single argument:

        output_file: this is the path to the file to output the CSV data to

    Optional arguments:

        ensemble_list: a JSON string representation of a dictionary where keys are
            UUID strings and values are ensemble names.
            A single * can be used to export all ensembles

        design_matrix: a path to a file containing the design matrix

    The script also looks for default values for output path and design matrix
    path to present in the GUI. These can be specified with DATA_KW keyword in
    the config file:

        DATA_KW <CSV_OUTPUT_PATH> {some path}
        DATA_KW <DESIGN_MATRIX_PATH> {some path}
    """

    @staticmethod
    def getName() -> str:
        return "CSV Export"

    @staticmethod
    def getDescription() -> str:
        return (
            "Export GenKW, design matrix, misfit data "
            "and summary data into a single CSV file."
        )

    def run(
        self,
        storage: Storage,
        workflow_args: Sequence[str],
    ) -> str:
        output_file = workflow_args[0]
        ensemble_data_as_json = None if len(workflow_args) < 2 else workflow_args[1]
        design_matrix_path = None if len(workflow_args) < 3 else workflow_args[2]
        drop_const_cols = False if len(workflow_args) < 5 else workflow_args[4]

        ensemble_data_as_dict = (
            json.loads(ensemble_data_as_json) if ensemble_data_as_json else {}
        )

        # Use the keys (UUIDs as strings) to get ensembles
        ensembles: list[Ensemble] = []
        for ensemble_id in ensemble_data_as_dict:
            assert storage is not None
            ensemble = storage.get_ensemble(ensemble_id)
            ensembles.append(ensemble)

        if design_matrix_path is not None:
            if not os.path.exists(design_matrix_path):
                raise UserWarning("The design matrix file does not exist!")

            if not os.path.isfile(design_matrix_path):
                raise UserWarning("The design matrix is not a file!")

        data = pd.DataFrame()

        for ensemble in ensembles:
            if not ensemble.has_data():
                raise UserWarning(
                    f"The ensemble '{ensemble.name}' does not have any data!"
                )

            ensemble_data = ensemble.load_all_gen_kw_data()

            if design_matrix_path is not None:
                design_matrix_data = loadDesignMatrix(design_matrix_path)
                if not design_matrix_data.empty:
                    ensemble_data = ensemble_data.join(design_matrix_data, how="outer")

            misfit_data = LibresFacade.load_all_misfit_data(ensemble)
            if not misfit_data.empty:
                ensemble_data = ensemble_data.join(misfit_data, how="outer")
            realizations = ensemble.get_realization_list_with_responses()

            try:
                summary_data = ensemble.load_responses("summary", tuple(realizations))
            except (KeyError, ValueError):
                summary_data = pl.DataFrame({})

            if not summary_data.is_empty():
                ensemble_data = ensemble_data.join(
                    summary_data.to_pandas(), how="outer"
                )
            else:
                ensemble_data["Date"] = None
                ensemble_data.set_index(["Date"], append=True, inplace=True)

            ensemble_data["Iteration"] = ensemble.iteration
            ensemble_data["Ensemble"] = ensemble.name
            ensemble_data.set_index(
                ["Ensemble", "Iteration"], append=True, inplace=True
            )

            data = pd.concat([data, ensemble_data])
        if not data.empty:
            data = data.reorder_levels(["Realization", "Iteration", "Date", "Ensemble"])
        if drop_const_cols:
            data = data.loc[:, (data != data.iloc[0]).any()]

        data.to_csv(output_file, float_format="%.6f")

        export_info = (
            f"Exported {len(data.index)} rows and {len(data.columns)} "
            f"columns to {output_file}."
        )
        return export_info
