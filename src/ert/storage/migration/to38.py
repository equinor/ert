from __future__ import annotations

import json
import logging
import uuid as _uuid
from pathlib import Path

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
    ensembles_dir = path / "ensembles"
    if not ensembles_dir.exists():
        return

    for ens_dir in ensembles_dir.iterdir():
        if not ens_dir.is_dir():
            continue

        for dataframe_name in _BATCH_DATAFRAME_NAMES:
            parquet_file = ens_dir / f"{dataframe_name}.parquet"
            if not parquet_file.exists():
                continue

            blob_dir = ens_dir / "blobs"
            blob_dir.mkdir(parents=True, exist_ok=True)

            data = parquet_file.read_bytes()
            uri = f"{_uuid.uuid4().hex[:8]}.blob"
            blob_data = {
                "uri": uri,
                "file_size": len(data),
                "file_type": "application/parquet",
                "name": dataframe_name,
                "blob_info": {
                    "blob_type": "everest_batch_data",
                    "dataframe_name": dataframe_name,
                },
            }

            (blob_dir / uri).write_bytes(data)
            (blob_dir / f"{uri}.json").write_text(
                json.dumps(blob_data, indent=2), encoding="utf-8"
            )
            parquet_file.unlink()
            logger.info("Moved %s into blob %s", parquet_file, uri)


def migrate(path: Path) -> None:
    _move_batch_dataframes_into_blobs(path)
