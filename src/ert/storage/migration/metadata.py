from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def migrate(path: Path) -> None:
    for experiment in path.glob("experiments/*"):
        if not (experiment / "metadata.json").exists():
            with open(experiment / "metadata.json", "w", encoding="utf-8") as fout:
                fout.write(json.dumps({}))
