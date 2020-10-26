from pydantic import BaseModel
from typing import Dict, List, Tuple
from collections import defaultdict

from ert_shared.ensemble_evaluator.entity import identifiers as ids
from ert_shared.ensemble_evaluator.entity.tool import (
    recursive_update,
    get_real_id,
    get_stage_id,
    get_step_id,
    get_job_id,
)

_FM_TYPE_EVENT_TO_STATUS = {
    ids.EVTYPE_FM_STAGE_WAITING: "waiting",
    ids.EVTYPE_FM_STAGE_PENDING: "pending",
    ids.EVTYPE_FM_STAGE_RUNNING: "running",
    ids.EVTYPE_FM_STAGE_FAILURE: "failure",
    ids.EVTYPE_FM_STAGE_SUCCESS: "success",
    ids.EVTYPE_FM_STAGE_UNKNOWN: "unknown",
    ids.EVTYPE_FM_STEP_START: "start",
    ids.EVTYPE_FM_STEP_FAILURE: "failure",
    ids.EVTYPE_FM_STEP_SUCCESS: "success",
    ids.EVTYPE_FM_JOB_START: "start",
    ids.EVTYPE_FM_JOB_RUNNING: "running",
    ids.EVTYPE_FM_JOB_SUCCESS: "success",
    ids.EVTYPE_FM_JOB_FAILURE: "failure",
}


class PartialSnapshot:
    def __init__(self, status=None):
        self._data = {"reals": {}}

    def update_status(self, status):
        self._data["status"] = status

    def update_real(self, real_id, active=None):
        if real_id not in self._data["reals"]:
            self._data["reals"][real_id] = {"stages": {}}
        real = self._data["reals"][real_id]

        if active is not None:
            real["active"] = active
        return real

    def update_stage(self, real_id, stage_id, status=None):
        real = self.update_real(real_id)
        if stage_id not in real["stages"]:
            real["stages"][stage_id] = {"steps": {}}
        stage = real["stages"][stage_id]

        if status is not None:
            stage["status"] = status
        return stage

    def update_step(self, real_id, stage_id, step_id, status=None):
        stage = self.update_stage(real_id, stage_id)
        if step_id not in stage["steps"]:
            stage["steps"][step_id] = {"jobs": {}}
        step = stage["steps"][step_id]

        if status is not None:
            step["status"] = status
        return step

    def update_job(self, real_id, stage_id, step_id, job_id, status=None, data=None):
        step = self.update_step(real_id, stage_id, step_id)
        if job_id not in step["jobs"]:
            step["jobs"][job_id] = {}
        job = step["jobs"][job_id]

        if status is not None:
            job["status"] = status

        if data is not None:
            if "data" not in job:
                job["data"] = {}
            job["data"].update(data)

        return job

    def to_dict(self):
        return self._data

    @classmethod
    def from_cloudevent(cls, event):
        snapshot = cls()
        e_type = event["type"]
        e_source = event["source"]
        status = _FM_TYPE_EVENT_TO_STATUS[e_type]
        if e_type in ids.EVGROUP_FM_STAGE:
            snapshot.update_stage(
                get_real_id(e_source), get_stage_id(e_source), status=status
            )
        elif e_type in ids.EVGROUP_FM_STEP:
            snapshot.update_step(
                get_real_id(e_source),
                get_stage_id(e_source),
                get_step_id(e_source),
                status=status,
            )
        elif e_type in ids.EVGROUP_FM_JOB:
            snapshot.update_job(
                get_real_id(e_source),
                get_stage_id(e_source),
                get_step_id(e_source),
                get_job_id(e_source),
                status=status,
            )
        else:
            raise ValueError()
        return snapshot


class Snapshot:
    def __init__(self, input_dict):
        self._data = input_dict

    def merge_event(self, event):
        recursive_update(self._data, event.to_dict())

    def to_dict(self):
        return self._data

    def get_status(self):
        return self._data["status"]

    def get_real(self, real_id):
        if real_id not in self._data["reals"]:
            raise ValueError(f"No realization with id {real_id}")
        return self._data["reals"][real_id]

    def get_stage(self, real_id, stage_id):
        real = self.get_real(real_id)
        stages = real["stages"]
        if stage_id not in stages:
            raise ValueError(f"No stage with id {stage_id} in {real_id}")
        return stages[stage_id]

    def get_step(self, real_id, stage_id, step_id):
        stage = self.get_stage(real_id, stage_id)
        steps = stage["steps"]
        if step_id not in steps:
            raise ValueError(f"No step with id {step_id} in {stage_id}")
        return steps[step_id]

    def get_job(self, real_id, stage_id, step_id, job_id):
        step = self.get_step(real_id, stage_id, step_id)
        jobs = step["jobs"]
        if job_id not in jobs:
            raise ValueError(f"No job with id {job_id} in {step_id}")
        return jobs[job_id]


class _JobDetails(BaseModel):
    job_id: Tuple[str, str, str]
    depends: List[Tuple[str, str, str]]


class _Job(BaseModel):
    status: str
    data: Dict


class _Step(BaseModel):
    status: str
    jobs: Dict[str, _Job] = {}


class _Stage(BaseModel):
    status: str
    steps: Dict[str, _Step] = {}


class _Realization(BaseModel):
    active: bool
    stages: Dict[str, _Stage] = {}


class _SnapshotDict(BaseModel):
    status: str
    reals: Dict[str, _Realization] = {}
    forward_model: List[_JobDetails]


class SnapshotBuilder(BaseModel):
    stages: Dict[str, _Stage] = {}
    forward_model: List[_JobDetails] = []

    def build(self, real_ids, status):
        top = _SnapshotDict(status=status, forward_model=self.forward_model)
        for r_id in real_ids:
            top.reals[r_id] = _Realization(active=True, stages=self.stages)
        return Snapshot(top.dict())

    def add_stage(self, stage_id, status):
        self.stages[stage_id] = _Stage(status=status)
        return self

    def add_step(self, stage_id, step_id, status):
        stage = self.stages[stage_id]
        stage.steps[step_id] = _Step(status=status)
        return self

    def add_job(self, stage_id, step_id, job_id, status, data, depends=None):
        stage = self.stages[stage_id]
        step = stage.steps[step_id]
        step.jobs[job_id] = _Job(status=status, data=data)
        if depends is not None:
            self.forward_model.append(
                _JobDetails(job_id=(stage_id, step_id, job_id), depends=depends)
            )
        return self
