import importlib.util
import sys
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional, Sequence

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
    fm_steps: Dict[str, FMStepSnapshot] = {}
    metadata: Dict[str, Any] = {}

    def build(
        self,
        real_ids: Sequence[str],
        status: Optional[str],
        exec_hosts: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
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
                ),
            )
        return snapshot

    def add_fm_step(
        self,
        fm_step_id: str,
        index: str,
        name: Optional[str],
        status: Optional[str],
        current_memory_usage: Optional[str] = None,
        max_memory_usage: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        stdout: Optional[str] = None,
        stderr: Optional[str] = None,
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
