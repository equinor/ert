from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

info = "Remove is_improvement flag from ensembles"


def migrate(path: Path) -> None:
    ensemble_dir = path / "ensembles"
    if not ensemble_dir.exists():
        return
    for ens_dir in ensemble_dir.iterdir():
        index_file = ens_dir / "index.json"
        if not index_file.exists():
            continue

        index_data = json.loads(index_file.read_text(encoding="utf-8"))

        if "is_improvement" in index_data:
            index_data.pop("is_improvement")
            index_file.write_text(json.dumps(index_data, indent=2), encoding="utf-8")
            logger.info("Removed is_improvement flag from %s", index_file)
