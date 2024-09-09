from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional, Sequence

from pydantic import BaseModel

from ert.ensemble_evaluator.snapshot import (
    ForwardModel,
    RealizationSnapshot,
    Snapshot,
    _filter_nones,
)


class SnapshotBuilder(BaseModel):
    forward_models: Dict[str, ForwardModel] = {}
    metadata: Dict[str, Any] = {}

    def build(
        self,
        real_ids: Sequence[str],
        status: Optional[str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Snapshot:
        snapshot = Snapshot()
        snapshot._ensemble_state = status
        snapshot._metadata = self.metadata

        for r_id in real_ids:
            snapshot.add_realization(
                r_id,
                RealizationSnapshot(
                    active=True,
                    forward_models=deepcopy(self.forward_models),
                    start_time=start_time,
                    end_time=end_time,
                    status=status,
                ),
            )
        return snapshot

    def add_forward_model(
        self,
        forward_model_id: str,
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
        self.forward_models[forward_model_id] = _filter_nones(
            ForwardModel(
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
