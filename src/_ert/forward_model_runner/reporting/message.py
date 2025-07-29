import dataclasses
from datetime import datetime as dt
from typing import TYPE_CHECKING, Literal, NotRequired, Self

import psutil
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from _ert.forward_model_runner.forward_model_step import ForwardModelStep


class Manifest(TypedDict):
    type: Literal["file"]
    path: str
    error: NotRequired[str]
    md5sum: NotRequired[str]


STEP_EXIT_FAILED_STRING_TEMPLATE = """Step {step_name} FAILED with code {exit_code}
----------------------------------------------------------
Error message: {error_message}
----------------------------------------------------------
"""


@dataclasses.dataclass
class ProcessTreeStatus:
    """Holds processtree information that can be represented as a line of CSV data"""

    timestamp: str = ""
    fm_step_id: int | None = None
    fm_step_name: str | None = None

    # Memory unit is bytes
    rss: int | None = None
    max_rss: int | None = None
    free: int | None = None

    cpu_seconds: float = 0.0

    oom_score: int | None = None

    def __post_init__(self) -> None:
        self.timestamp = dt.now().isoformat()
        self.free = psutil.virtual_memory().available

    def __repr__(self) -> str:
        return ",".join([str(value) for value in dataclasses.astuple(self)]).replace(
            "None", ""
        )

    def csv_header(self) -> str:
        return ",".join(dataclasses.asdict(self).keys())


class _MetaMessage(type):
    def __repr__(cls) -> str:
        return f"MessageType<{cls.__name__}>"


class Message(metaclass=_MetaMessage):
    def __init__(self, step: "ForwardModelStep | None" = None) -> None:
        self.timestamp = dt.now()
        self.step = step
        self.error_message: str | None = None

    def __repr__(self) -> str:
        return type(self).__name__

    def with_error(self, message: str) -> Self:
        self.error_message = message
        return self

    def success(self) -> bool:
        return self.error_message is None


# manager level messages


class Init(Message):
    def __init__(
        self,
        steps: list["ForwardModelStep"],
        run_id: str | None,
        ert_pid: str | None,
        ens_id: str | None = None,
        real_id: int | None = None,
        experiment_id: str | None = None,
    ) -> None:
        super().__init__()
        self.steps = steps
        self.run_id = run_id
        self.ert_pid = ert_pid
        self.experiment_id = experiment_id
        self.ens_id = ens_id
        self.real_id = real_id


class Finish(Message):
    def __init__(self) -> None:
        super().__init__()


class Start(Message):
    def __init__(self, fm_step: "ForwardModelStep") -> None:
        super().__init__(fm_step)


class Running(Message):
    def __init__(
        self, fm_step: "ForwardModelStep", memory_status: ProcessTreeStatus
    ) -> None:
        super().__init__(fm_step)
        self.memory_status = memory_status


class Exited(Message):
    def __init__(
        self, fm_step: "ForwardModelStep | None", exit_code: int | None
    ) -> None:
        super().__init__(fm_step)
        self.exit_code = exit_code


class Checksum(Message):
    def __init__(self, checksum_dict: dict[str, "Manifest"], run_path: str) -> None:
        super().__init__()
        self.data = checksum_dict
        self.run_path = run_path
