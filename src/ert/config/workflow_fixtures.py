from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qtpy.QtWidgets import QWidget
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from ert.config import ESSettings
    from ert.config.analysis_config import UpdateSettings
    from ert.runpaths import Runpaths
    from ert.storage import Ensemble, Storage


class WorkflowFixtures(TypedDict, total=False):
    ensemble: Ensemble | None
    storage: Storage
    random_seed: int | None
    reports_dir: str
    observation_settings: UpdateSettings
    es_settings: ESSettings
    run_paths: Runpaths
    workflow_args: list[Any]
    parent: QWidget | None
