from __future__ import annotations

import io
from pathlib import Path

import polars as pl

from .migration_utils import ensemble_dirs, write_parquet_blob

info = "Migrate observation_scaling_factors.parquet to blob storage"


def _migrate_scaling_factors(path: Path) -> None:
    for ens_dir in ensemble_dirs(path):
        old_file = ens_dir / "observation_scaling_factors.parquet"
        if not old_file.exists():
            continue

        try:
            df = pl.read_parquet(old_file)
        except Exception:
            continue

        buf = io.BytesIO()
        df.write_parquet(buf)

        num_groups = df["input_group"].n_unique() if "input_group" in df.columns else 1

        write_parquet_blob(
            ens_dir,
            buf.getvalue(),
            name="scaling_factors",
            blob_info={
                "blob_type": "scaling_factors",
                "update_algorithm": "ensemble_smoother",
                "num_observations": len(df),
                "num_groups": num_groups,
            },
        )
        old_file.unlink()


def migrate(path: Path) -> None:
    _migrate_scaling_factors(path)
