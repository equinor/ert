from __future__ import annotations

import logging
from pathlib import Path

from .migration_utils import ensemble_dirs, read_json, write_json

logger = logging.getLogger(__name__)

info = "Add parameter_group_sizes field to matrix blob metadata"


def _add_parameter_group_sizes(path: Path) -> None:
    for ens_dir in ensemble_dirs(path):
        blob_dir = ens_dir / "blobs"
        if not blob_dir.exists():
            continue

        for meta_file in blob_dir.glob("*.blob.json"):
            try:
                meta = read_json(meta_file)
            except Exception:
                continue

            blob_info = meta.get("blob_info")
            if blob_info is None:
                continue

            if blob_info.get("blob_type") != "matrix":
                continue

            if "parameter_group_sizes" in blob_info:
                continue

            blob_info["parameter_group_sizes"] = {}
            write_json(meta_file, meta)
            logger.info("Added parameter_group_sizes to %s", meta_file)


def migrate(path: Path) -> None:
    _add_parameter_group_sizes(path)
