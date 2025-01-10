import importlib.util
import sys
from collections.abc import Sequence
from copy import deepcopy
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from ert.ensemble_evaluator.snapshot import (
    EnsembleSnapshot,
    FMStepSnapshot,
    RealizationSnapshot,
    _filter_nones,
)


def import_from_location(name, location):
    spec = importlib.util.spec_from_file_location(name, location)
    if spec is None:
        raise ImportError(f"Could not find {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    if spec.loader is None:
        raise ImportError(f"No loader for {name}")
    spec.loader.exec_module(module)
    return module


class SnapshotBuilder(BaseModel):
    fm_steps: dict[str, FMStepSnapshot] = {}
    metadata: dict[str, Any] = {}

    def build(
        self,
        real_ids: Sequence[str],
        status: str | None,
        exec_hosts: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        message: str | None = None,
    ) -> EnsembleSnapshot:
        snapshot = EnsembleSnapshot()
        snapshot._ensemble_state = status
        snapshot._metadata = self.metadata

        for r_id in real_ids:
            snapshot.add_realization(
                r_id,
                RealizationSnapshot(
                    active=True,
                    fm_steps=deepcopy(self.fm_steps),
                    start_time=start_time,
                    end_time=end_time,
                    exec_hosts=exec_hosts,
                    status=status,
                    message=message,
                ),
            )
        return snapshot

    def add_fm_step(
        self,
        fm_step_id: str,
        index: str,
        name: str | None,
        status: str | None,
        current_memory_usage: str | None = None,
        max_memory_usage: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        stdout: str | None = None,
        stderr: str | None = None,
    ) -> "SnapshotBuilder":
        self.fm_steps[fm_step_id] = _filter_nones(
            FMStepSnapshot(
                status=status,
                index=index,
                start_time=start_time,
                end_time=end_time,
                name=name,
                stdout=stdout,
                stderr=stderr,
                current_memory_usage=current_memory_usage,
                max_memory_usage=max_memory_usage,
            )
        )
        return self
