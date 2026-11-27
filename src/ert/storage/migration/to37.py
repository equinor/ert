from __future__ import annotations

import logging
from pathlib import Path

from .migration_utils import ensemble_dirs, read_json, write_json

logger = logging.getLogger(__name__)

info = "Remove is_improvement flag from ensembles"


def migrate(path: Path) -> None:
    for ens_dir in ensemble_dirs(path):
        index_file = ens_dir / "index.json"
        if not index_file.exists():
            continue

        index_data = read_json(index_file)

        if "is_improvement" in index_data:
            index_data.pop("is_improvement")
            write_json(index_file, index_data)
            logger.info("Removed is_improvement flag from %s", index_file)
