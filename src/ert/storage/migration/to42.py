from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

info = (
    "Remove cached rho matrix blobs, which were labelled by observation name "
    "instead of observation id (response key and index)"
)


def _remove_rho_matrix_blobs(path: Path) -> None:
    experiments_dir = path / "experiments"
    if not experiments_dir.exists():
        return

    for exp_dir in experiments_dir.iterdir():
        blob_dir = exp_dir / "blobs"
        if not blob_dir.is_dir():
            continue
        resolved_blob_dir = blob_dir.resolve()

        for json_path in blob_dir.glob("*.json"):
            blob_data = json.loads(json_path.read_text(encoding="utf-8"))
            if blob_data.get("blob_info", {}).get("blob_type") != "rho_matrix":
                continue

            uri = blob_data.get("uri")
            if isinstance(uri, str):
                blob_path = (resolved_blob_dir / uri).resolve()
                if blob_path.is_relative_to(resolved_blob_dir):
                    blob_path.unlink(missing_ok=True)
                else:
                    logger.warning(
                        "Skipping rho matrix blob outside blob directory: %s", uri
                    )
            json_path.unlink()
            logger.info("Removed rho matrix blob metadata %s", json_path)


def migrate(path: Path) -> None:
    _remove_rho_matrix_blobs(path)
