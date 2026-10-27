from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

info = "Move SCALAR.parquet into ensemble blobs"


def _move_scalar_parquet_into_blobs(path: Path) -> None:
    ensembles_dir = path / "ensembles"
    if not ensembles_dir.exists():
        return

    for ens_dir in ensembles_dir.iterdir():
        parquet_file = ens_dir / "SCALAR.parquet"
        if not ens_dir.is_dir() or not parquet_file.exists():
            continue

        blob_dir = ens_dir / "blobs"
        blob_dir.mkdir(parents=True, exist_ok=True)

        data = parquet_file.read_bytes()
        uri = f"{uuid.uuid4().hex[:8]}.blob"
        blob_data = {
            "uri": uri,
            "file_size": len(data),
            "file_type": "application/parquet",
            "name": "SCALAR",
            "blob_info": {"blob_type": "scalars"},
        }

        (blob_dir / uri).write_bytes(data)
        (blob_dir / f"{uri}.json").write_text(
            json.dumps(blob_data, indent=2), encoding="utf-8"
        )
        parquet_file.unlink()
        logger.info("Moved %s into blob %s", parquet_file, uri)


def migrate(path: Path) -> None:
    _move_scalar_parquet_into_blobs(path)
