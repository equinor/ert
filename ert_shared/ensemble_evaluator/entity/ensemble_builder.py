from ert_shared.ensemble_evaluator.entity import ensemble
from ert_shared.ensemble_evaluator.entity import identifiers as ids
from ert_shared.ensemble_evaluator.entity.ensemble import _Ensemble


class _Job:
    pass


class _LegacyJob(_Job):
    def __init__(self):
        self._name = None
        self._id = None
        self._executable = None
        self._target_file = None
        self._error_file = None
        self._start_file = None
        self._stdout = None
        self._stderr = None
        self._stdin = None
        self._argList = None
        self._environment = None
        self._exec_env = None
        self._license_path = None
        self._max_running_minutes = None
        self._max_running = None
        self._min_arg = None
        self._arg_types = None
        self._max_arg = None
        self._status = "unknown"

    @staticmethod
    def from_dict(data):
        j = _LegacyJob()
        j._name = data.get(ids.FM_JOB_ATTR_NAME)
        j._executable = data.get(ids.FM_JOB_ATTR_EXECUTABLE)
        j._target_file = data.get(ids.FM_JOB_ATTR_TARGET_FILE)
        j._error_file = data.get(ids.FM_JOB_ATTR_ERROR_FILE)
        j._start_file = data.get(ids.FM_JOB_ATTR_START_FILE)
        j._stdout = data.get(ids.FM_JOB_ATTR_STDOUT)
        j._stderr = data.get(ids.FM_JOB_ATTR_STDERR)
        j._stdin = data.get(ids.FM_JOB_ATTR_STDIN)
        j._argList = data.get(ids.FM_JOB_ATTR_ARGLIST)
        j._environment = data.get(ids.FM_JOB_ATTR_ENVIRONMENT)
        j._exec_env = data.get(ids.FM_JOB_ATTR_EXEC_ENV)
        j._license_path = data.get(ids.FM_JOB_ATTR_LICENSE_PATH)
        j._max_running_minutes = data.get(ids.FM_JOB_ATTR_MAX_RUNNING_MINUTES)
        j._max_running = data.get(ids.FM_JOB_ATTR_MAX_RUNNING)
        j._min_arg = data.get(ids.FM_JOB_ATTR_MIN_ARG)
        j._arg_types = data.get(ids.FM_JOB_ATTR_ARG_TYPES)
        j._max_arg = data.get(ids.FM_JOB_ATTR_MAX_ARG)
        j._status = data.get(ids.FM_JOB_ATTR_STATUS, j._status)
        return j

    def to_dict(self):
        return {
            ids.FM_JOB_ATTR_NAME: self._name,
            ids.FM_JOB_ATTR_EXECUTABLE: self._executable,
            ids.FM_JOB_ATTR_TARGET_FILE: self._target_file,
            ids.FM_JOB_ATTR_ERROR_FILE: self._error_file,
            ids.FM_JOB_ATTR_START_FILE: self._start_file,
            ids.FM_JOB_ATTR_STDOUT: self._stdout,
            ids.FM_JOB_ATTR_STDERR: self._stderr,
            ids.FM_JOB_ATTR_STDIN: self._stdin,
            ids.FM_JOB_ATTR_ARGLIST: self._argList,
            ids.FM_JOB_ATTR_ENVIRONMENT: self._environment,
            ids.FM_JOB_ATTR_EXEC_ENV: self._exec_env,
            ids.FM_JOB_ATTR_LICENSE_PATH: self._license_path,
            ids.FM_JOB_ATTR_MAX_RUNNING_MINUTES: self._max_running_minutes,
            ids.FM_JOB_ATTR_MAX_RUNNING: self._max_running,
            ids.FM_JOB_ATTR_MIN_ARG: self._min_arg,
            ids.FM_JOB_ATTR_ARG_TYPES: self._arg_types,
            ids.FM_JOB_ATTR_MAX_ARG: self._max_arg,
            ids.FM_JOB_ATTR_STATUS: self._status,
        }

    def id(self):
        return self._id

    def build(self):
        return self.to_dict()


class _Step:
    def __init__(self):
        self._id = None
        self._jobs = None
        self.reset()

    def reset(self):
        self._jobs = []
        self._depends = set()

    def depends(self, step):
        if not isinstance(step, _Step):
            raise TypeError("expected to depend on Step, got {step}")
        self._depends.add(step)
        return self

    def add(self, job):
        if not isinstance(job, _Job):
            raise TypeError("expected Job got {job}")
        job._id = len(self._jobs)
        self._jobs.append(job)
        return self

    def id(self):
        return self._id

    def build(self):
        return {"jobs": {str(job.id()): job.build() for job in self._jobs}}


class _Stage:
    def __init__(self):
        self._id = None
        self._steps = None
        self.depends = None
        self.reset()

    def reset(self):
        self._steps = []
        self._depends = set()

    def depends(self, stage):
        if not isinstance(stage, _Stage):
            raise TypeError("expected to depend on Stage, got {stage}")
        self._depends.add(stage)
        return self

    def id(self):
        return self._id

    def add(self, step):
        if not isinstance(step, _Step):
            raise TypeError("expected Step got {step}")
        step._id = len(self._steps)
        self._steps.append(step)
        return self

    def build(self):
        return {"steps": {str(step.id()): step.build() for step in self._steps}}


class Builder:
    def __init__(self):
        self._stages = None
        self._ensemble_size = None
        self.reset()

    def reset(self):
        self._ensemble_size = 0
        self._stages = []

    def add_stage(self, stage):
        if not isinstance(stage, _Stage):
            raise TypeError("expected Stage got {stage}")
        stage._id = len(self._stages)
        self._stages.append(stage)
        return self

    def set_ensemble_size(self, size):
        self._ensemble_size = size
        return self

    def build(self):
        return _Ensemble(
            {
                "status": "unknown",
                "reals": {
                    str(iens): {
                        "stages": {str(stage.id()): stage.build() for stage in self._stages}
                    }
                    for iens in range(0, self._ensemble_size)
                }
            }
        )


class LegacyBuilder(Builder):
    def __init__(self):
        super().__init__()
        self._only_step = _Step()
        self._only_stage = _Stage().add(self._only_step)

        self.add_stage(self._only_stage)

    def add_job(self, data):
        job = _LegacyJob.from_dict(data)
        self._only_step.add(job)
        return self
