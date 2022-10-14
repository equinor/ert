from itertools import chain
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

from ert._c_wrappers.enkf import RunArg

from ._function_task import FunctionTask
from ._io_ import IO, DummyIO, DummyIOBuilder, InputBuilder, IOBuilder, OutputBuilder
from ._io_map import _stage_transmitter_mapping
from ._job import BaseJob, FunctionJob, JobBuilder, LegacyJob, LegacyJobBuilder, UnixJob
from ._stage import Stage, StageBuilder
from ._unix_task import UnixTask

SOURCE_TEMPLATE_STEP = "/step/{step_id}"

callback = Callable[[List[Any]], Union[bool, Tuple[Any, str]]]


class Step(Stage):
    def __init__(  # pylint: disable=too-many-arguments
        self,
        id_: str,
        inputs: Sequence[IO],
        outputs: Sequence[IO],
        jobs: Sequence[BaseJob],
        name: str,
        source: str,
    ) -> None:
        super().__init__(id_, name, inputs, outputs)
        self.jobs = jobs
        self._source = source

    def source(self) -> str:
        return self._source


class UnixStep(Step):
    def __init__(  # pylint: disable=too-many-arguments
        self,
        id_: str,
        inputs: Sequence[IO],
        outputs: Sequence[IO],
        jobs: Sequence[BaseJob],
        name: str,
        source: str,
        run_path: Optional[Path] = None,
    ) -> None:
        super().__init__(id_, inputs, outputs, jobs, name, source)
        self._run_path = run_path

    def get_task(
        self,
        output_transmitters: _stage_transmitter_mapping,
        ens_id: str,
        *args: Any,
        **kwargs: Any,
    ) -> UnixTask:
        return UnixTask(
            self, output_transmitters, ens_id, *args, run_path=self._run_path, **kwargs
        )


class FunctionStep(Step):
    def get_task(
        self,
        output_transmitters: _stage_transmitter_mapping,
        ens_id: str,
        *args: Any,
        **kwargs: Any,
    ) -> FunctionTask:
        return FunctionTask(self, output_transmitters, ens_id, *args, **kwargs)


class LegacyStep(Step):  # pylint: disable=too-many-instance-attributes
    def __init__(  # pylint: disable=too-many-arguments
        self,
        id_: str,
        inputs: Sequence[IO],
        outputs: Sequence[IO],
        jobs: Sequence[LegacyJob],
        name: str,
        ee_url: str,
        source: str,
        max_runtime: Optional[int],
        callback_arguments: Optional[Sequence[Any]],
        done_callback: callback,
        exit_callback: callback,
        num_cpu: Optional[int],
        run_path: Path,
        job_script: str,
        job_name: str,
        run_arg: RunArg,
    ) -> None:
        super().__init__(id_, inputs, outputs, jobs, name, source)
        if max_runtime is not None and max_runtime <= 0:
            raise ValueError(f"{self} needs positive max_runtime")
        self.max_runtime = max_runtime
        self.callback_arguments = callback_arguments
        self.done_callback = done_callback
        self.exit_callback = exit_callback
        self.num_cpu = num_cpu
        self.run_path = run_path
        self.job_script = job_script
        self.job_name = job_name
        self.run_arg = run_arg


class StepBuilder(StageBuilder):  # pylint: disable=too-many-instance-attributes
    def __init__(self) -> None:
        super().__init__()
        self._jobs: List[Union[JobBuilder, LegacyJobBuilder]] = []
        self._type: Optional[str] = None
        self._parent_source: Optional[str] = None

        # legacy parts
        self._max_runtime: Optional[int] = None
        self._callback_arguments: Optional[Sequence[Any]] = None
        self._done_callback: Optional[callback] = None
        self._exit_callback: Optional[callback] = None
        self._num_cpu: Optional[int] = None
        self._run_path: Optional[Path] = None
        self._job_script: Optional[str] = None
        self._job_name: Optional[str] = None
        self._run_arg: Optional[RunArg] = None

    def set_id(self, id_: str) -> "StepBuilder":
        super().set_id(id_)
        return self

    def set_name(self, name: str) -> "StepBuilder":
        super().set_name(name)
        return self

    def add_output(self, output: IOBuilder) -> "StepBuilder":
        if isinstance(output, InputBuilder):
            raise TypeError(f"adding input '{output._name}' as output")
        super().add_output(output)
        return self

    def add_input(self, input_: IOBuilder) -> "StepBuilder":
        if isinstance(input_, OutputBuilder):
            raise TypeError(f"adding output '{input_._name}' as input")
        super().add_input(input_)
        return self

    def set_type(self, type_: str) -> "StepBuilder":
        self._type = type_
        return self

    def set_parent_source(self, source: str) -> "StepBuilder":
        self._parent_source = source
        return self

    def add_job(self, job: Union[JobBuilder, LegacyJobBuilder]) -> "StepBuilder":
        self._jobs.append(job)
        return self

    def set_max_runtime(self, max_runtime: Optional[int]) -> "StepBuilder":
        if max_runtime and max_runtime > 0:
            self._max_runtime = max_runtime
        return self

    def set_callback_arguments(
        self, callback_arguments: Sequence[Any]
    ) -> "StepBuilder":
        self._callback_arguments = callback_arguments
        return self

    def set_done_callback(self, done_callback: callback) -> "StepBuilder":
        self._done_callback = done_callback
        return self

    def set_exit_callback(self, exit_callback: callback) -> "StepBuilder":
        self._exit_callback = exit_callback
        return self

    def set_num_cpu(self, num_cpu: int) -> "StepBuilder":
        self._num_cpu = num_cpu
        return self

    def set_run_path(self, run_path: Path) -> "StepBuilder":
        self._run_path = run_path
        return self

    def set_job_script(self, job_script: str) -> "StepBuilder":
        self._job_script = job_script
        return self

    def set_job_name(self, job_name: str) -> "StepBuilder":
        self._job_name = job_name
        return self

    def set_run_arg(self, run_arg: RunArg) -> "StepBuilder":
        self._run_arg = run_arg
        return self

    def set_dummy_io(self) -> "StepBuilder":
        self.add_input(DummyIOBuilder())
        self.add_output(DummyIOBuilder())
        return self

    # pylint: disable=too-many-branches
    def build(self) -> Union[LegacyStep, FunctionStep, UnixStep]:
        stage = super().build()
        if not self._parent_source:
            raise ValueError(f"need parent_source for {self._name}")
        source = self._parent_source + SOURCE_TEMPLATE_STEP.format(step_id=stage.id_)

        # only legacy has _run_arg, so assume it is a legacy step
        if self._run_arg:
            legacy_jobs: List[LegacyJob] = []
            for builder in self._jobs:
                if isinstance(builder, LegacyJobBuilder):
                    legacy_jobs.append(builder.set_parent_source(source).build())
            if self._callback_arguments is None:
                raise ValueError(f"{self} needs callback_arguments")
            if self._done_callback is None:
                raise ValueError(f"{self} needs done_callback")
            if self._exit_callback is None:
                raise ValueError(f"{self} needs exit_callback")
            if self._num_cpu is None:
                raise ValueError(f"{self} needs num_cpu")
            if self._run_path is None:
                raise ValueError(f"{self} needs run_path")
            if self._job_script is None:
                raise ValueError(f"{self} needs job_script")
            if self._job_name is None:
                raise ValueError(f"{self} needs job_name")
            if self._run_arg is None:
                raise ValueError(f"{self} needs run_arg")

            return LegacyStep(
                stage.id_,
                stage.inputs,
                stage.outputs,
                legacy_jobs,
                self._name,
                "",  # ee_url
                "",  # source
                self._max_runtime,
                self._callback_arguments,
                self._done_callback,
                self._exit_callback,
                self._num_cpu,
                self._run_path,
                self._job_script,
                self._job_name,
                self._run_arg,
            )
        if self._type == "function":
            function_jobs: List[FunctionJob] = []
            for builder in self._jobs:
                job = builder.set_parent_source(source).build()
                if isinstance(job, FunctionJob):
                    function_jobs.append(job)

            return FunctionStep(
                stage.id_,
                stage.inputs,
                stage.outputs,
                function_jobs,
                stage.name,
                source,
            )
        elif self._type == "unix":
            for io_ in chain(stage.inputs, stage.outputs):
                # dummy io are used for legacy ensembles can not transform
                if isinstance(io_, DummyIO):
                    continue

                if not io_.transformation:
                    raise ValueError(
                        f"cannot build {self._type} step: {io_} has no transformation"
                    )

            unix_jobs: List[UnixJob] = []
            for builder in self._jobs:
                job = builder.set_parent_source(source).build()
                if isinstance(job, UnixJob):
                    unix_jobs.append(job)

            return UnixStep(
                stage.id_,
                stage.inputs,
                stage.outputs,
                unix_jobs,
                stage.name,
                source,
                run_path=self._run_path,
            )
        else:
            raise ValueError("Unexpected type while building step: {self._type}")
