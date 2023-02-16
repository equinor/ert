import uuid
from typing import Any, Callable, List, Optional

from typing_extensions import Self

from ert.job_queue import ExtJob

SOURCE_TEMPLATE_JOB = "/job/{job_id}/index/{job_index}"


class BaseJob:
    def __init__(self, id_: str, index: str, name: str, source: str) -> None:
        self.id_ = id_
        self.name = name
        self.index = index
        self._source = source

    def source(self) -> str:
        return self._source


class LegacyJob(BaseJob):
    def __init__(
        self,
        id_: str,
        index: str,
        name: str,
        ext_job: ExtJob,
    ) -> None:
        super().__init__(id_, index, name, "")  # no step_source needed for legacy (pt.)
        self.ext_job = ext_job


class BaseJobBuilder:
    def __init__(self) -> None:
        self._id: Optional[str] = None
        self._index: Optional[str] = None
        self._name: Optional[str] = None
        self._parent_source: Optional[str] = None

    def set_id(self, id_: str) -> Self:
        self._id = id_
        return self

    def set_parent_source(self, source: str) -> Self:
        self._parent_source = source
        return self

    def set_name(self, name: str) -> Self:
        self._name = name
        return self

    def set_index(self, index: str) -> Self:
        self._index = index
        return self

    def build(self) -> BaseJob:
        raise NotImplementedError("cannot build basejob")


# pylint: disable=too-many-instance-attributes
class LegacyJobBuilder(BaseJobBuilder):
    def __init__(self) -> None:
        super().__init__()
        self._ext_job: Optional[ExtJob] = None
        self._script: Optional[str] = None
        self._run_path: Optional[str] = None
        self._num_cpu: Optional[int] = None
        self._status_file: Optional[str] = None
        self._exit_file: Optional[str] = None
        self._done_callback_function: Optional[Callable[[List[Any]], bool]] = None
        self._exit_callback_function: Optional[Callable[[List[Any]], bool]] = None
        self._callback_arguments: Optional[List[Any]] = None
        self._max_runtime: Optional[int] = None

    def set_ext_job(self, ext_job: ExtJob) -> Self:
        self._ext_job = ext_job
        return self

    def build(self) -> LegacyJob:
        if self._id is None:
            self._id = str(uuid.uuid4())
        if self._index is None:
            raise ValueError("Job needs an index")
        if self._name is None:
            raise ValueError("legacy job must have name")
        if self._ext_job is None:
            raise ValueError(f"legacy job {self._name} must have ExtJob")
        return LegacyJob(self._id, self._index, self._name, self._ext_job)
