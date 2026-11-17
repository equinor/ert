from __future__ import annotations

import logging
from pathlib import Path

from .migration_utils import ensemble_dirs, read_json, resolve_blob_path

logger = logging.getLogger(__name__)

info = (
    "Remove stored EnIF Kalman gain (K) blobs computed with the prior "
    "precision matrix instead of the posterior one"
)


def _remove_enif_kalman_blobs(path: Path) -> None:
    for ens_dir in ensemble_dirs(path):
        blob_dir = ens_dir / "blobs"
        if not blob_dir.is_dir():
            continue

        for json_path in blob_dir.glob("*.json"):
            blob_data = read_json(json_path)
            blob_info = blob_data.get("blob_info", {})
            if (
                blob_data.get("name") == "K"
                and blob_info.get("blob_type") == "matrix"
                and blob_info.get("update_algorithm") == "enif"
            ):
                uri = blob_data.get("uri")
                if not isinstance(uri, str):
                    logger.warning(
                        "Skipping EnIF K blob without string URI: %s", json_path
                    )
                    continue

                blob_path = resolve_blob_path(blob_dir, uri)
                if blob_path is None:
                    logger.warning(
                        "Skipping EnIF K blob outside blob directory: %s", uri
                    )
                    continue

                blob_path.unlink(missing_ok=True)
                json_path.unlink()
                logger.info("Removed wrong EnIF K blob %s", blob_path)


def migrate(path: Path) -> None:
    _remove_enif_kalman_blobs(path)
