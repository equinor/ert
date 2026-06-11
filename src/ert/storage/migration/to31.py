from __future__ import annotations

import io
import json
import uuid
from pathlib import Path

import polars as pl

info = "Migrate observation_scaling_factors.parquet to blob storage"


def _migrate_scaling_factors(path: Path) -> None:
    ensembles_dir = path / "ensembles"
    if not ensembles_dir.exists():
        return

    for ens_dir in ensembles_dir.iterdir():
        if not ens_dir.is_dir():
            continue

        old_file = ens_dir / "observation_scaling_factors.parquet"
        if not old_file.exists():
            continue

        try:
            df = pl.read_parquet(old_file)
        except Exception:
            continue

        blob_dir = ens_dir / "blobs"
        blob_dir.mkdir(parents=True, exist_ok=True)

        blob_id = uuid.uuid4().hex[:8]
        blob_file = blob_dir / f"{blob_id}.blob"
        blob_meta_file = blob_dir / f"{blob_id}.blob.json"

        buf = io.BytesIO()
        df.write_parquet(buf)
        blob_bytes = buf.getvalue()

        num_groups = df["input_group"].n_unique() if "input_group" in df.columns else 1

        blob_meta = {
            "uri": f"{blob_id}.blob",
            "file_size": len(blob_bytes),
            "file_type": "application/parquet",
            "name": "scaling_factors",
            "blob_info": {
                "blob_type": "scaling_factors",
                "update_algorithm": "ensemble_smoother",
                "num_observations": len(df),
                "num_groups": num_groups,
            },
        }

        blob_file.write_bytes(blob_bytes)
        blob_meta_file.write_text(json.dumps(blob_meta, indent=2), encoding="utf-8")
        old_file.unlink()


def migrate(path: Path) -> None:
    _migrate_scaling_factors(path)
