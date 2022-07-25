from pathlib import Path
import pickle
import uuid
from typing import (
    Callable,
    List,
    Tuple,
    Any,
    Optional,
    TypeVar,
    Union,
)
from res.job_queue.ext_job import ExtJob


from ._template import _SOURCE_TEMPLATE_BASE, _SOURCE_TEMPLATE_JOB


_BaseJobBuilder_TV = TypeVar("_BaseJobBuilder_TV", bound="_BaseJobBuilder")

_callable_or_path = Union[bytes, Path]


class _BaseJob:
    def __init__(self, id_: str, index: str, name: str, source: str) -> None:
        self.id_ = id_
        self.name = name
        self.index = index
        self._source = source

    def source(self, ee_id: str) -> str:
        return self._source.format(ee_id=ee_id)


class _UnixJob(_BaseJob):
    def __init__(  # pylint: disable=too-many-arguments
        self,
        id_: str,
        index: str,
        name: str,
        step_source: str,
        executable: Path,
        args: Optional[Tuple[str, ...]] = None,
    ) -> None:
        super().__init__(id_, index, name, step_source)
        self.executable = executable
        self.args = args


class _FunctionJob(_BaseJob):
    # pylint: disable=too-many-arguments
    # index is required by job sorting for gui
    def __init__(
        self,
        id_: str,
        index: str,
        name: str,
        step_source: str,
        command: bytes,
    ) -> None:
        super().__init__(id_, index, name, step_source)
        self.command = command


class _LegacyJob(_BaseJob):
    def __init__(
        self,
        id_: str,
        index: str,
        name: str,
        ext_job: ExtJob,
    ) -> None:
        super().__init__(id_, index, name, "")  # no step_source needed for legacy (pt.)
        self.ext_job = ext_job


class _BaseJobBuilder:
    def __init__(self) -> None:
        self._id: Optional[str] = None
        self._index: Optional[str] = None
        self._name: Optional[str] = None
        self._parent_source: Optional[str] = None

    def set_id(self: _BaseJobBuilder_TV, id_: str) -> _BaseJobBuilder_TV:
        self._id = id_
        return self

    def set_parent_source(self: _BaseJobBuilder_TV, source: str) -> _BaseJobBuilder_TV:
        self._parent_source = source
        return self

    def set_name(self: _BaseJobBuilder_TV, name: str) -> _BaseJobBuilder_TV:
        self._name = name
        return self

    def set_index(self: _BaseJobBuilder_TV, index: str) -> _BaseJobBuilder_TV:
        self._index = index
        return self

    def build(self) -> _BaseJob:
        raise NotImplementedError("cannot build basejob")


class _JobBuilder(_BaseJobBuilder):
    def __init__(self) -> None:
        super().__init__()
        self._executable: Optional[_callable_or_path] = None
        self._args: Optional[Tuple[str, ...]] = None

    def set_executable(self, executable: _callable_or_path) -> "_JobBuilder":
        self._executable = executable
        return self

    def set_args(self, args: Tuple[str, ...]) -> "_JobBuilder":
        self._args = args
        return self

    def build(self) -> Union[_FunctionJob, _UnixJob]:
        if self._id is None:
            self._id = str(uuid.uuid4())
        if self._index is None:
            raise ValueError("Job needs an index")
        if self._name is None:
            raise ValueError("job needs a name")
        if self._parent_source is None:
            raise ValueError("job need source of parent")
        source = (
            _SOURCE_TEMPLATE_BASE
            + self._parent_source
            + _SOURCE_TEMPLATE_JOB.format(job_id=self._id, job_index=self._index)
        )
        try:
            cmd_is_callable = isinstance(self._executable, bytes) and callable(
                pickle.loads(self._executable)
            )
        except TypeError:
            cmd_is_callable = False
        if cmd_is_callable:
            if self._args:
                raise ValueError(
                    "callable executable does not take args, use inputs instead"
                )

            assert isinstance(self._executable, bytes)  # mypy
            return _FunctionJob(
                self._id, self._index, self._name, source, self._executable
            )

        if not isinstance(self._executable, Path):
            raise TypeError(f"executable in job '{self._name}' should be a {Path}")
        return _UnixJob(
            self._id, self._index, self._name, source, self._executable, self._args
        )


# pylint: disable=too-many-instance-attributes
class _LegacyJobBuilder(_BaseJobBuilder):
    def __init__(self) -> None:
        super().__init__()
        self._ext_job: Optional[ExtJob] = None
        self._script: Optional[str] = None
        self._run_path: Optional[str] = None
        self._num_cpu: Optional[int] = None
        self._status_file: Optional[str] = None
        self._ok_file: Optional[str] = None
        self._exit_file: Optional[str] = None
        self._done_callback_function: Optional[Callable[[List[Any]], bool]] = None
        self._exit_callback_function: Optional[Callable[[List[Any]], bool]] = None
        self._callback_arguments: Optional[List[Any]] = None
        self._max_runtime: Optional[int] = None

    def set_ext_job(self, ext_job: ExtJob) -> "_LegacyJobBuilder":
        self._ext_job = ext_job
        return self

    def build(self) -> _LegacyJob:
        if self._id is None:
            self._id = str(uuid.uuid4())
        if self._index is None:
            raise ValueError("Job needs an index")
        if self._name is None:
            raise ValueError("legacy job must have name")
        if self._ext_job is None:
            raise ValueError(f"legacy job {self._name} must have ExtJob")
        return _LegacyJob(self._id, self._index, self._name, self._ext_job)
