from __future__ import annotations

import logging
from pathlib import Path

from .migration_utils import experiment_dirs, read_json, resolve_blob_path

logger = logging.getLogger(__name__)

info = (
    "Remove cached rho matrix blobs, which were labelled by observation name "
    "instead of observation id (response key and index)"
)


def _remove_rho_matrix_blobs(path: Path) -> None:
    for exp_dir in experiment_dirs(path):
        blob_dir = exp_dir / "blobs"
        if not blob_dir.is_dir():
            continue

        for json_path in blob_dir.glob("*.json"):
            blob_data = read_json(json_path)
            if blob_data.get("blob_info", {}).get("blob_type") != "rho_matrix":
                continue

            uri = blob_data.get("uri")
            if isinstance(uri, str):
                blob_path = resolve_blob_path(blob_dir, uri)
                if blob_path is not None:
                    blob_path.unlink(missing_ok=True)
                else:
                    logger.warning(
                        "Skipping rho matrix blob outside blob directory: %s", uri
                    )
            json_path.unlink()
            logger.info("Removed rho matrix blob metadata %s", json_path)


def migrate(path: Path) -> None:
    _remove_rho_matrix_blobs(path)
