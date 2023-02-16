from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence

from ._io_ import IO, DummyIOBuilder, InputBuilder, IOBuilder, OutputBuilder
from ._job import BaseJob, LegacyJob, LegacyJobBuilder
from ._stage import Stage, StageBuilder

SOURCE_TEMPLATE_STEP = "/step/{step_id}"
if TYPE_CHECKING:
    from ert._c_wrappers.enkf import RunArg
    from ert.callbacks import Callback, CallbackArgs


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
        callback_arguments: CallbackArgs,
        done_callback: Callback,
        exit_callback: Callback,
        num_cpu: int,
        run_path: Path,
        job_script: str,
        job_name: str,
        run_arg: "RunArg",
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
        self._jobs: List[LegacyJobBuilder] = []
        self._type: Optional[str] = None
        self._parent_source: Optional[str] = None

        # legacy parts
        self._max_runtime: Optional[int] = None
        self._callback_arguments: Optional[CallbackArgs] = None
        self._done_callback: Optional[Callback] = None
        self._exit_callback: Optional[Callback] = None
        self._num_cpu: Optional[int] = None
        self._run_path: Optional[Path] = None
        self._job_script: Optional[str] = None
        self._job_name: Optional[str] = None
        self._run_arg: Optional["RunArg"] = None

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

    def add_job(self, job: LegacyJobBuilder) -> "StepBuilder":
        self._jobs.append(job)
        return self

    def set_max_runtime(self, max_runtime: Optional[int]) -> "StepBuilder":
        if max_runtime and max_runtime > 0:
            self._max_runtime = max_runtime
        return self

    def set_callback_arguments(self, callback_arguments: CallbackArgs) -> "StepBuilder":
        self._callback_arguments = callback_arguments
        return self

    def set_done_callback(self, done_callback: Callback) -> "StepBuilder":
        self._done_callback = done_callback
        return self

    def set_exit_callback(self, exit_callback: Callback) -> "StepBuilder":
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

    def set_run_arg(self, run_arg: "RunArg") -> "StepBuilder":
        self._run_arg = run_arg
        return self

    def set_dummy_io(self) -> "StepBuilder":
        self.add_input(DummyIOBuilder())
        self.add_output(DummyIOBuilder())
        return self

    # pylint: disable=too-many-branches
    def build(self) -> LegacyStep:
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
        else:
            raise ValueError("Unexpected type while building step: {self._type}")
