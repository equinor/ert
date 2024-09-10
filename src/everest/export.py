import os
import re
import sys
from typing import Any, Dict, List, Optional, Set

import pandas as pd
from pandas import DataFrame
from seba_sqlite.snapshot import SebaSnapshot

from ert.storage import open_storage
from everest.config import EverestConfig
from everest.strings import STORAGE_DIR

if sys.version_info < (3, 11):
    from enum import Enum

    class StrEnum(str, Enum):
        pass

else:
    from enum import StrEnum


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


def filter_data(data: DataFrame, keyword_filters: Set[str]):
    filtered_columns = []

    for col in data.columns:
        for expr in keyword_filters:
            expr = expr.replace("*", ".*")
            if re.match(expr, col) is not None:
                filtered_columns.append(col)

    return data[filtered_columns]


def _valid_batches(batches: List[int], config: EverestConfig):
    snapshot = SebaSnapshot(config.optimization_output_dir).get_snapshot(
        filter_out_gradient=False, batches=None
    )
    available_batches = {data.batch for data in snapshot.simulation_data}
    valid_batches = [batch for batch in batches if batch in available_batches]
    return valid_batches


def _metadata(config: EverestConfig):
    discard_gradient = True
    discard_rejected = True
    batches = None

    if config.export is not None:
        if config.export.discard_gradient is not None:
            discard_gradient = config.export.discard_gradient

        if config.export.discard_rejected is not None:
            discard_rejected = config.export.discard_rejected

        if config.export.batches:
            # If user defined batches to export in the conf file, ignore previously
            # discard gradient and discard rejected flags if defined and true
            discard_rejected = False
            discard_gradient = False
            batches = config.export.batches

    snapshot = SebaSnapshot(config.optimization_output_dir).get_snapshot(
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

        md_row: Dict[str, Any] = {
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
                            {"gradient-{}-{}".format(function, control): gradient_value}
                        )
            else:
                print(
                    "Batch {} has no available optimization data".format(
                        md_row[MetaDataColumnNames.BATCH]
                    )
                )
        metadata.append(md_row)

    return metadata


def get_internalized_keys(config: EverestConfig, batch_ids: Optional[Set[int]] = None):
    if batch_ids is None:
        metadata = _metadata(config)
        batch_ids = {data[MetaDataColumnNames.BATCH] for data in metadata}
    internal_keys: Set = set()
    with open_storage(config.storage_dir, "r") as storage:
        for batch_id in batch_ids:
            case_name = f"batch_{batch_id}"
            experiments = [*storage.experiments]
            assert len(experiments) == 1
            experiment = experiments[0]

            ensemble = experiment.get_ensemble_by_name(case_name)
            if not internal_keys:
                internal_keys = set(ensemble.get_summary_keyset())
            else:
                internal_keys = internal_keys.intersection(
                    set(ensemble.get_summary_keyset())
                )

    return internal_keys


def validate_export(config: EverestConfig):
    """
    Checks for possible errors when attempting to export current optimization
    case.
    :param config: Everest config
    :return: List of error messages
    """
    export_ecl = True
    export_errors: List[str] = []
    if config.export is None:
        return export_errors, export_ecl

    # TODO turn into attr accessor when ExplicitNone & everlint is phased out
    user_def_batches = config.export.batches
    if user_def_batches:
        valid_batches = _valid_batches(user_def_batches, config)
        for batch in user_def_batches:
            if batch not in valid_batches:
                export_errors.append(
                    "Batch {} not found in optimization "
                    "results. Skipping for current export."
                    "".format(batch)
                )
        user_def_batches = valid_batches
        config.export.batches = user_def_batches

    if user_def_batches == []:
        export_ecl = False
        export_errors.append(
            "No batches selected for export. "
            "Only optimization data will be exported."
        )
        return export_errors, export_ecl

    if not config.model.data_file:
        export_ecl = False
        export_errors.append(
            "No data file found in config." "Only optimization data will be exported."
        )

    user_def_kw = config.export.keywords
    if user_def_kw == []:
        export_ecl = False
        export_errors.append(
            "No eclipse keywords selected for export. Only"
            " optimization data will be exported."
        )

    # If no user defined keywords are present it is no longer possible to check
    # availability in internal storage
    if user_def_kw is None:
        return export_errors, export_ecl

    internal_keys = get_internalized_keys(
        config, set(user_def_batches) if user_def_batches else None
    )

    extra_keys = set(user_def_kw).difference(set(internal_keys))
    if extra_keys:
        export_ecl = False
        export_errors.append(
            f"Non-internalized ecl keys selected for export '{' '.join(extra_keys)}'."
            " in order to internalize missing keywords "
            f"run 'everest load {config.config_file}'. "
            "Only optimization data will be exported."
        )

    return export_errors, export_ecl


def export(config: EverestConfig, export_ecl=True, progress_callback=lambda _: None):
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
    if config.export is not None:
        if config.export.skip_export or config.export.batches == []:
            return pd.DataFrame([])

        ecl_keywords = config.export.keywords

    metadata = _metadata(config)
    data_file = config.model.data_file
    if data_file is None or not export_ecl:
        return pd.DataFrame(metadata)

    data = _load_simulation_data(config, metadata, progress_callback=progress_callback)

    if ecl_keywords is not None:
        keywords = tuple(ecl_keywords)
        # NOTE: Some of these keywords are necessary for a success full export,
        # should not leave this to the user..
        keywords += tuple(pd.DataFrame(metadata).columns)
        keywords += tuple(MetaDataColumnNames.get_all())
        keywords_set = set(keywords)
        data = filter_data(data, keywords_set)

    return data


def _load_simulation_data(
    config: EverestConfig, metadata: List[dict], progress_callback=lambda _: None
):
    """Export simulations to a pandas DataFrame
    @config the case configuration
    @tags is a one ora a list of dictionaries. Keys from the dictionary become
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
    assert config.output_dir is not None  # avoiding mypy error
    ens_path = os.path.join(config.output_dir, STORAGE_DIR)
    with open_storage(ens_path, "r") as storage:
        # pylint: disable=unnecessary-lambda-assignment
        def load_batch_by_id():
            case_name = f"batch_{batch}"
            experiments = [*storage.experiments]

            # Always assume 1 experiment per simulation/enspath, never multiple
            assert len(experiments) == 1
            experiment = experiments[0]

            ensemble = experiment.get_ensemble_by_name(case_name)
            return ensemble.load_all_summary_data()

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
