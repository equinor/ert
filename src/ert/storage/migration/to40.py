from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

info = (
    "Remove stored EnIF Kalman gain (K) blobs computed with the prior "
    "precision matrix instead of the posterior one"
)


def _resolve_blob_path(blob_dir: Path, uri: str) -> Path | None:
    resolved_blob_dir = blob_dir.resolve()
    blob_path = (resolved_blob_dir / uri).resolve()
    try:
        blob_path.relative_to(resolved_blob_dir)
    except ValueError:
        logger.warning("Skipping EnIF K blob outside blob directory: %s", uri)
        return None
    return blob_path


def _remove_enif_kalman_blobs(path: Path) -> None:
    ensembles_dir = path / "ensembles"
    if not ensembles_dir.exists():
        return

    for ens_dir in ensembles_dir.iterdir():
        blob_dir = ens_dir / "blobs"
        if not blob_dir.is_dir():
            continue

        for json_path in blob_dir.glob("*.json"):
            blob_data = json.loads(json_path.read_text(encoding="utf-8"))
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

                blob_path = _resolve_blob_path(blob_dir, uri)
                if blob_path is None:
                    continue

                blob_path.unlink(missing_ok=True)
                json_path.unlink()
                logger.info("Removed wrong EnIF K blob %s", blob_path)


def migrate(path: Path) -> None:
    _remove_enif_kalman_blobs(path)
