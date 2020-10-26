import copy
from ert_shared.ensemble_evaluator.entity.snapshot import SnapshotBuilder


class _IO:
    def __init__(self, name, type_, path):
        if not name:
            raise ValueError(f"{self} needs name")
        self._name = name
        self._type = type_
        self._path = path


class _IOBuilder:
    _concrete_cls = None

    def __init__(self):
        self._name = None
        self._type = None
        self._path = None

    def reset(self):
        self._name = None
        self._type = None
        self._path = None
        return self

    def set_name(self, name):
        self._name = name
        return self

    def set_type(self, type_):
        self._type = type_
        return self

    def set_path(self, path):
        self._path = path
        return self

    def build(self):
        if self._concrete_cls is None:
            raise TypeError("cannot build _IO")
        return self._concrete_cls(self._name, self._type, self._path)


class _IODummy(_IO):
    pass


class _IODummyBuilder:
    _concrete_cls = _IODummy

    def build(_):
        return _IO("dummy i/o", "dummy", "")


class _Input(_IO):
    pass


class _InputBuilder(_IOBuilder):
    _concrete_cls = _Input


def create_input_builder():
    return _InputBuilder()


class _Output(_IO):
    pass


class _OutputBuilder(_IOBuilder):
    _concrete_cls = _Output


def create_output_builder():
    return _OutputBuilder()


class _BaseJobBuilder:
    def __init__(self):
        self._id = None
        self._name = None

    def reset(self):
        self._id = None
        self._name = None
        return self

    def set_id(self, id_):
        self._id = id_
        return self

    def set_name(self, name):
        self._name = name
        return self

    def build(_):
        raise NotImplementedError("cannot build basejob")


class _ScriptJobBuilder(_BaseJobBuilder):
    def __init__(self):
        super().__init__()
        self._executable = None
        self._args = None
        self.reset()

    def reset(self):
        super().reset()
        self._executable = None
        self._args = tuple()
        return self

    def set_executable(self, executable):
        self._executable = executable
        return self

    def set_args(self, args):
        self._args = args
        return self

    def build(self):
        return _ScriptJob(self._id, self._name, self._executable, self._args)


def create_script_job_builder():
    return _ScriptJobBuilder()


class _BaseJob:
    def __init__(self, id_, name):
        if id_ is None:
            raise ValueError(f"{self} need id")
        if name is None:
            raise ValueError(f"{self} need name")
        self._id = id_
        self._name = name

    def get_id(self):
        return self._id

    def get_name(self):
        return self._name


class _ScriptJob(_BaseJob):
    def __init__(
        self,
        id_,
        name,
        executable,
        args,
    ):
        super().__init__(id_, name)
        if not executable:
            raise ValueError(f"{self} need executable")
        if args is None or not isinstance(args, tuple):
            raise TypeError(f"{args} have to be tuple")
        self._executable = executable
        self._args = args

    def get_executable(self):
        return self._executable

    def get_args(self):
        return self._args


class _Step:
    def __init__(self, id_, inputs, outputs, jobs):
        if id_ is None:
            raise ValueError(f"{self} needs id")
        if not inputs:
            raise ValueError(f"{self} needs input")
        if not outputs:
            raise ValueError(f"{self} needs output")
        if not jobs:
            raise ValueError(f"{self} needs jobs")

        self._id = id_
        self._inputs = inputs
        self._ouputs = outputs
        self._jobs = jobs

    def get_id(self):
        return self._id

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def get_jobs(self):
        return self._jobs


class _StepBuilder:
    def __init__(self):
        self._jobs = None
        self._id = None
        self._inputs = None
        self._outputs = None
        self.reset()

    def reset(self):
        self._jobs = []
        self._inputs = []
        self._outputs = []
        self._id = None
        return self

    def set_id(self, id_):
        self._id = id_
        return self

    def add_job(self, job):
        self._jobs.append(job)
        return self

    def add_output(self, output):
        self._outputs.append(output)
        return self

    def add_input(self, input):
        self._inputs.append(input)
        return self

    def build(self):
        jobs = [builder.build() for builder in self._jobs]
        inputs = [builder.build() for builder in self._inputs]
        outputs = [builder.build() for builder in self._outputs]
        return _Step(self._id, inputs, outputs, jobs)

    def set_dummy_io(self):
        self.add_input(_IODummyBuilder())
        self.add_output(_IODummyBuilder())
        return self


def create_step_builder():
    return _StepBuilder()


class _Stage:
    def __init__(self, id_, steps, status):
        if id_ is None:
            raise ValueError(f"{self} needs id")
        if not steps:
            raise ValueError(f"{self} needs steps")
        if not status:
            raise ValueError(f"{self} needs status")

        self._id = id_
        self._steps = steps
        self._status = status

    def get_id(self):
        return self._id

    def get_steps(self):
        return self._steps

    def get_status(self):
        return self._status


class _StageBuilder:
    def __init__(self):
        self._steps = None
        self._id = None
        self._status = None
        self.reset()

    def reset(self):
        self._steps = []
        self._id = None
        self._status = None
        return self

    def set_id(self, id_):
        self._id = id_
        return self

    def set_status(self, status):
        self._status = status
        return self

    def add_step(self, step):
        self._steps.append(step)
        return self

    def build(self):
        steps = [builder.build() for builder in self._steps]
        return _Stage(self._id, steps, self._status)


def create_stage_builder():
    return _StageBuilder()


class _RealizationBuilder:
    def __init__(self):
        self._stages = None
        self._active = None
        self._iens = None
        self.reset()

    def reset(self):
        self._stages = []
        self._active = None
        self._iens = None
        return self

    def active(self, active):
        self._active = active
        return self

    def add_stage(self, stage):
        self._stages.append(stage)
        return self

    def set_iens(self, iens):
        self._iens = iens
        return self

    def build(self):
        stages = [builder.build() for builder in self._stages]
        return _Realization(self._iens, stages, self._active)


def create_realization_builder():
    return _RealizationBuilder()


class _Realization:
    def __init__(self, iens, stages, active):
        if iens is None:
            raise ValueError(f"{self} needs iens")
        if not stages:
            raise ValueError(f"{self} needs stages")
        if active is None:
            raise ValueError(f"{self} needs to be set either active or not")

        self._iens = iens
        self._stages = stages
        self._active = active

    def get_iens(self):
        return self._iens

    def get_stages(self):
        return self._stages

    def is_active(self):
        return self._active

    def set_active(self, active):
        self._active = active


class _EnsembleBuilder:
    def __init__(self):
        self._reals = None
        self._size = None
        self.reset()

    def reset(self):
        self._reals = []
        self._size = 0
        return self

    def add_realization(self, real):
        self._reals.append(real)
        return self

    def set_ensemble_size(self, size):
        """Duplicate the ensemble members that existed at build time so as to
        get the desired state."""
        self._size = size
        return self

    @staticmethod
    def from_legacy(run_context, forward_model):
        builder = _EnsembleBuilder()

        for iens in range(0, len(run_context)):
            step = create_step_builder().set_id(0)

            for index in range(0, len(forward_model)):
                ext_job = forward_model.iget_job(index)
                step.add_job(
                    create_script_job_builder()
                    .set_executable(ext_job.get_executable())
                    .set_args(tuple(arg for arg in ext_job.get_arglist()))
                    .set_id(index)
                    .set_name(ext_job.name())
                ).set_dummy_io()

            builder.add_realization(
                create_realization_builder()
                .add_stage(
                    create_stage_builder()
                    .add_step(step)
                    .set_id(0)
                    .set_status("unknown")
                )
                .set_iens(iens)
                .active(run_context.is_active(iens))
            )

        return builder

    def build(self):
        # duplicate the original reals
        # XXX: this is likely very stupid
        orig_len = len(self._reals)
        for i in range(orig_len, self._size):
            real = copy.deepcopy(self._reals[i % orig_len])
            real.set_iens(i)
            self._reals.append(real)

        reals = [builder.build() for builder in self._reals]
        return _Ensemble(reals)


def create_ensemble_builder():
    return _EnsembleBuilder()


def create_ensemble_builder_from_legacy(run_context, forward_model):
    return _EnsembleBuilder.from_legacy(run_context, forward_model)


class _Ensemble:
    def __init__(self, reals):
        self._reals = reals

    def __repr__(self):
        return f"Ensemble with {len(self._reals)} members"

    def evaluate(self, host, port):
        pass

    def get_reals(self):
        return self._reals

    def forward_model_description(self):
        builder = SnapshotBuilder()
        real = self._reals[0]
        for stage in real.get_stages():
            builder.add_stage(str(stage.get_id()), stage.get_status())
            for step in stage.get_steps():
                builder.add_step(str(stage.get_id()), str(step.get_id()), "unknown")
                for job in step.get_jobs():
                    builder.add_job(
                        str(stage.get_id()),
                        str(step.get_id()),
                        str(job.get_id()),
                        job.get_name(),
                        "unknown",
                        {},
                    )
        return builder.build([str(real.get_iens()) for real in self._reals], "unknown")
