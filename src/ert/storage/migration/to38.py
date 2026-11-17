from __future__ import annotations

import logging
from pathlib import Path

from .migration_utils import ensemble_dirs, write_parquet_blob

logger = logging.getLogger(__name__)

info = "Move everest batch dataframes into ensemble blobs"

_BATCH_DATAFRAME_NAMES = (
    "batch_objectives",
    "batch_constraints",
    "batch_bound_constraint_violations",
    "batch_input_constraint_violations",
    "batch_output_constraint_violations",
    "batch_objective_gradient",
    "batch_constraint_gradient",
)


def _move_batch_dataframes_into_blobs(path: Path) -> None:
    for ens_dir in ensemble_dirs(path):
        for dataframe_name in _BATCH_DATAFRAME_NAMES:
            parquet_file = ens_dir / f"{dataframe_name}.parquet"
            if not parquet_file.exists():
                continue

            uri = write_parquet_blob(
                ens_dir,
                parquet_file.read_bytes(),
                name=dataframe_name,
                blob_info={
                    "blob_type": "everest_batch_data",
                    "dataframe_name": dataframe_name,
                },
            )
            parquet_file.unlink()
            logger.info("Moved %s into blob %s", parquet_file, uri)


def migrate(path: Path) -> None:
    _move_batch_dataframes_into_blobs(path)
