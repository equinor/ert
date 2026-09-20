from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

info = "Add parameter_group_sizes field to matrix blob metadata"


def _add_parameter_group_sizes(path: Path) -> None:
    ensembles_dir = path / "ensembles"
    if not ensembles_dir.exists():
        return

    for ens_dir in ensembles_dir.iterdir():
        if not ens_dir.is_dir():
            continue

        blob_dir = ens_dir / "blobs"
        if not blob_dir.exists():
            continue

        for meta_file in blob_dir.glob("*.blob.json"):
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
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
            meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            logger.info("Added parameter_group_sizes to %s", meta_file)


def migrate(path: Path) -> None:
    _add_parameter_group_sizes(path)
