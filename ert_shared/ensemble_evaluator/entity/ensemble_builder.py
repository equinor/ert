from pydantic import BaseModel
from typing import Dict


class _Job(BaseModel):
    status: str
    data: Dict
    current_memory_usage: int


class _Step(BaseModel):
    status: str
    jobs: Dict[int, _Job]


class _Stage(BaseModel):
    status: str
    steps: Dict[int, _Step]


class _Realization(BaseModel):
    status: str
    stages: Dict[int, _Stage]


class _SnapshotDict(BaseModel):
    reals: Dict[int, _Realization] = []


class SnapshotBuilder(BaseModel):
    stages: Dict[int, _Stage] = []

    def build(self, real_ids):
        top = _SnapshotDict()
        for id in real_ids:
            top.reals.add(id, _Realization( status="unknown", stages=self.stages))
        return top.dict()

    def add_stage(self, stage_id, status):
        self.stages.add(stage_id, _Stage(status=status))
        return self

    def add_step(self, stage_id, step_id, status):
        stage = self.stages[stage_id]
        stage.steps.add(step_id, _Step(status=status))
        return self

    def add_job(self, stage_id, step_id, job_id, status, data):
        stage = self.stages[stage_id]
        step = stage.steps[step_id]
        step.jobs.add(job_id, _Job(status=status, data=data))
