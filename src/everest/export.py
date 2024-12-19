import os
import re
from enum import StrEnum
from typing import Any

import pandas as pd
from pandas import DataFrame
from seba_sqlite.snapshot import SebaSnapshot

from ert.storage import open_storage
from everest.config import ExportConfig
from everest.strings import STORAGE_DIR


class MetaDataColumnNames(StrEnum):
    # NOTE: Always add a new column name to the list below!
    BATCH = "batch"
    REALIZATION = "realization"
    REALIZATION_WEIGHT = "realization_weight"
    SIMULATION = "simulation"
    IS_GRADIENT = "is_gradient"
    SUCCESS = "success"
    START_TIME = "start_time"
    END_TIME = "end_time"
    SIM_AVERAGED_OBJECTIVE = "sim_avg_obj"
    REAL_AVERAGED_OBJECTIVE = "real_avg_obj"
    SIMULATED_DATE = "sim_date"
    INCREASED_MERIT = "increased_merit"

    @classmethod
    def get_all(cls):
        return [
            cls.BATCH,
            cls.REALIZATION,
            cls.REALIZATION_WEIGHT,
            cls.SIMULATION,
            cls.IS_GRADIENT,
            cls.SUCCESS,
            cls.START_TIME,
            cls.END_TIME,
            cls.SIM_AVERAGED_OBJECTIVE,
            cls.REAL_AVERAGED_OBJECTIVE,
            cls.SIMULATED_DATE,
            cls.INCREASED_MERIT,
        ]


def filter_data(data: DataFrame, keyword_filters: set[str]):
    filtered_columns = []

    for col in data.columns:
        for expr in keyword_filters:
            expr = expr.replace("*", ".*")
            if re.match(expr, col) is not None:
                filtered_columns.append(col)

    return data[filtered_columns]


def available_batches(optimization_output_dir: str) -> set[int]:
    snapshot = SebaSnapshot(optimization_output_dir).get_snapshot(
        filter_out_gradient=False, batches=None
    )
    return {data.batch for data in snapshot.simulation_data}


def export_metadata(config: ExportConfig | None, optimization_output_dir: str):
    discard_gradient = True
    discard_rejected = True
    batches = None

    if config:
        if config.discard_gradient is not None:
            discard_gradient = config.discard_gradient

        if config.discard_rejected is not None:
            discard_rejected = config.discard_rejected

        if config.batches:
            # If user defined batches to export in the conf file, ignore previously
            # discard gradient and discard rejected flags if defined and true
            discard_rejected = False
            discard_gradient = False
            batches = config.batches

    snapshot = SebaSnapshot(optimization_output_dir).get_snapshot(
        filter_out_gradient=discard_gradient,
        batches=batches,
    )

    opt_data = snapshot.optimization_data_by_batch
    metadata = []
    for data in snapshot.simulation_data:
        # If export section not defined in the config file export only increased
        # merit non-gradient simulation results
        if (
            discard_rejected
            and data.batch in opt_data
            and opt_data[data.batch].merit_flag != 1
        ):
            continue

        md_row: dict[str, Any] = {
            MetaDataColumnNames.BATCH: data.batch,
            MetaDataColumnNames.SIM_AVERAGED_OBJECTIVE: data.sim_avg_obj,
            MetaDataColumnNames.IS_GRADIENT: data.is_gradient,
            MetaDataColumnNames.REALIZATION: int(data.realization),
            MetaDataColumnNames.START_TIME: data.start_time,
            MetaDataColumnNames.END_TIME: data.end_time,
            MetaDataColumnNames.SUCCESS: data.success,
            MetaDataColumnNames.REALIZATION_WEIGHT: data.realization_weight,
            MetaDataColumnNames.SIMULATION: int(data.simulation),
        }
        if data.objectives:
            md_row.update(data.objectives)
        if data.constraints:
            md_row.update(data.constraints)
        if data.controls:
            md_row.update(data.controls)

        if not md_row[MetaDataColumnNames.IS_GRADIENT]:
            if md_row[MetaDataColumnNames.BATCH] in opt_data:
                opt = opt_data[md_row[MetaDataColumnNames.BATCH]]
                md_row.update(
                    {
                        MetaDataColumnNames.REAL_AVERAGED_OBJECTIVE: opt.objective_value,
                        MetaDataColumnNames.INCREASED_MERIT: opt.merit_flag,
                    }
                )
                for function, gradients in opt.gradient_info.items():
                    for control, gradient_value in gradients.items():
                        md_row.update(
                            {f"gradient-{function}-{control}": gradient_value}
                        )
            else:
                print(
                    f"Batch {md_row[MetaDataColumnNames.BATCH]} has no available optimization data"
                )
        metadata.append(md_row)

    return metadata


def get_internalized_keys(
    config: ExportConfig,
    storage_path: str,
    optimization_output_path: str,
    batch_ids: set[int] | None = None,
):
    if batch_ids is None:
        metadata = export_metadata(config, optimization_output_path)
        batch_ids = {data[MetaDataColumnNames.BATCH] for data in metadata}
    internal_keys: set = set()
    with open_storage(storage_path, "r") as storage:
        for batch_id in batch_ids:
            experiments = [*storage.experiments]
            assert len(experiments) == 1
            experiment = experiments[0]

            ensemble = experiment.get_ensemble_by_name(f"batch_{batch_id}")
            if not internal_keys:
                internal_keys = set(
                    ensemble.experiment.response_type_to_response_keys["summary"]
                )
            else:
                internal_keys = internal_keys.intersection(
                    set(ensemble.experiment.response_type_to_response_keys["summary"])
                )

    return internal_keys


def check_for_errors(
    config: ExportConfig,
    optimization_output_path: str,
    storage_path: str,
    data_file_path: str | None,
):
    """
    Checks for possible errors when attempting to export current optimization
    case.
    """
    export_ecl = True
    export_errors: list[str] = []

    if config.batches:
        available_batches_ = available_batches(optimization_output_path)
        for batch in set(config.batches).difference(available_batches_):
            export_errors.append(
                f"Batch {batch} not found in optimization "
                "results. Skipping for current export."
            )
        config.batches = list(set(config.batches).intersection(available_batches_))

    if config.batches == []:
        export_errors.append(
            "No batches selected for export. "
            "Only optimization data will be exported."
        )
        return export_errors, False

    if not data_file_path:
        export_ecl = False
        export_errors.append(
            "No data file found in config." "Only optimization data will be exported."
        )

    # If no user defined keywords are present it is no longer possible to check
    # availability in internal storage
    if config.keywords is None:
        return export_errors, export_ecl

    if not config.keywords:
        export_ecl = False
        export_errors.append(
            "No eclipse keywords selected for export. Only"
            " optimization data will be exported."
        )

    internal_keys = get_internalized_keys(
        config=config,
        storage_path=storage_path,
        optimization_output_path=optimization_output_path,
        batch_ids=set(config.batches) if config.batches else None,
    )

    extra_keys = set(config.keywords).difference(set(internal_keys))
    if extra_keys:
        export_ecl = False
        export_errors.append(
            f"Non-internalized ecl keys selected for export '{' '.join(extra_keys)}'."
            " in order to internalize missing keywords "
            f"run 'everest load <config_file>'. "
            "Only optimization data will be exported."
        )

    return export_errors, export_ecl


def export_data(
    export_config: ExportConfig | None,
    output_dir: str,
    data_file: str | None,
    export_ecl=True,
    progress_callback=lambda _: None,
):
    """Export everest data into a pandas dataframe. If the config specifies
    a data_file and @export_ecl is True, simulation data is included. When
    exporting simulation data, only keywords matching elements in @ecl_keywords
    are exported. Note that wildcards are allowed.

    @progress_callback will be called with a number between 0 and 1 indicating
    the fraction of batches that has been loaded.
    """

    ecl_keywords = None
    # If user exports with a config file that has the SKIP_EXPORT
    # set to true export nothing
    if export_config is not None:
        if export_config.skip_export or export_config.batches == []:
            return pd.DataFrame([])

        ecl_keywords = export_config.keywords
    optimization_output_dir = os.path.join(
        os.path.abspath(output_dir), "optimization_output"
    )
    metadata = export_metadata(export_config, optimization_output_dir)
    if data_file is None or not export_ecl:
        return pd.DataFrame(metadata)

    data = load_simulation_data(
        output_path=output_dir,
        metadata=metadata,
        progress_callback=progress_callback,
    )

    if ecl_keywords is not None:
        keywords = tuple(ecl_keywords)
        # NOTE: Some of these keywords are necessary to export successfully,
        # we should not leave this to the user
        keywords += tuple(pd.DataFrame(metadata).columns)
        keywords += tuple(MetaDataColumnNames.get_all())
        keywords_set = set(keywords)
        data = filter_data(data, keywords_set)

    return data


def load_simulation_data(
    output_path: str, metadata: list[dict], progress_callback=lambda _: None
):
    """Export simulations to a pandas DataFrame
    @output_path optimization output folder path.
    @metadata is a one ora a list of dictionaries. Keys from the dictionary become
    columns in the resulting dataframe. The values from the dictionary are
    assigned to those columns for the corresponding simulation.
    If a column is defined for some simulations but not for others, the value
    for that column is set to NaN for simulations without it

    For instance, assume we have 2 simulations and
      tags = [ {'geoid': 0, 'sim': 'ro'},
               {'geoid': 2, 'sim': 'pi', 'best': True },
             ]
    And assume exporting each of the two simulations produces 3 rows.
    The resulting dataframe will be something like
        geoid  sim  best  data...
      0   0     ro        sim_0_row_0...
      1   0     ro        sim_0_row_1...
      2   0     ro        sim_0_row_2...
      3   2     pi  True  sim_1_row_0...
      4   2     pi  True  sim_2_row_0...
      5   2     pi  True  sim_3_row_0...
    """
    ens_path = os.path.join(output_path, STORAGE_DIR)
    with open_storage(ens_path, "r") as storage:
        # pylint: disable=unnecessary-lambda-assignment
        def load_batch_by_id():
            experiments = [*storage.experiments]

            # Always assume 1 experiment per simulation/enspath, never multiple
            assert len(experiments) == 1
            experiment = experiments[0]

            ensemble = experiment.get_ensemble_by_name(f"batch_{batch}")
            realizations = ensemble.get_realization_list_with_responses()
            try:
                df_pl = ensemble.load_responses("summary", tuple(realizations))
            except (ValueError, KeyError):
                return pd.DataFrame()
            df_pl = df_pl.pivot(
                on="response_key", index=["realization", "time"], sort_columns=True
            )
            df_pl = df_pl.rename({"time": "Date", "realization": "Realization"})
            return (
                df_pl.to_pandas()
                .set_index(["Realization", "Date"])
                .sort_values(by=["Date", "Realization"])
            )

        batches = {elem[MetaDataColumnNames.BATCH] for elem in metadata}
        batch_data = []
        for idx, batch in enumerate(batches):
            progress_callback(float(idx) / len(batches))
            batch_data.append(load_batch_by_id())
            batch_data[-1][MetaDataColumnNames.BATCH] = batch

    for b in batch_data:
        b.reset_index(inplace=True)
        b.rename(
            index=str,
            inplace=True,
            columns={
                "Realization": MetaDataColumnNames.SIMULATION,
                "Date": MetaDataColumnNames.SIMULATED_DATE,
            },
        )

    data = pd.concat(batch_data, ignore_index=True, sort=False)
    data = pd.merge(
        left=data,
        right=pd.DataFrame(metadata),
        on=[MetaDataColumnNames.BATCH, MetaDataColumnNames.SIMULATION],
        sort=False,
    )

    return data
