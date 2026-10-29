from __future__ import annotations

import json
from pathlib import Path

info = "Rename experiment status field from status to state"


def migrate(path: Path) -> None:
    experiments_dir = path / "experiments"
    if not experiments_dir.exists():
        return

    for exp_dir in experiments_dir.iterdir():
        index_path = exp_dir / "index.json"
        if not index_path.is_file():
            continue

        index = json.loads(index_path.read_text(encoding="utf-8"))
        status = index.get("status")
        if isinstance(status, dict) and "status" in status:
            status["state"] = status.pop("status")
            index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
