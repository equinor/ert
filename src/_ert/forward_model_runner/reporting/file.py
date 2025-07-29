import datetime
import logging
import os
import socket
import time
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from _ert.forward_model_runner.forward_model_step import ForwardModelStep
import orjson

from _ert.forward_model_runner.io import cond_unlink
from _ert.forward_model_runner.reporting.base import Reporter
from _ert.forward_model_runner.reporting.message import (
    STEP_EXIT_FAILED_STRING_TEMPLATE,
    Exited,
    Finish,
    Init,
    Message,
    Running,
    Start,
)
from _ert.forward_model_runner.util.data import (
    StepDict,
    StepStatus,
    create_step_dict,
    datetime_serialize,
)

TIME_FORMAT = "%H:%M:%S"
logger = logging.getLogger(__name__)
memory_logger = logging.getLogger("_ert.forward_model_memory_profiler")

LOG_file = "JOB_LOG"
ERROR_file = "ERROR"
STATUS_file = "STATUS"
OK_file = "OK"
STATUS_json = "status.json"


class StatusDict(TypedDict, total=False):
    run_id: str | None
    start_time: float | None
    end_time: float | None
    steps: list[StepDict]
    status: StepStatus


class File(Reporter):
    def __init__(self) -> None:
        self.status_dict: StatusDict = {}
        self.node = socket.gethostname()

    def report(self, msg: Message) -> None:
        fm_step_status: StepDict = {}

        if msg.step:
            logger.debug("Adding message step to status dictionary.")
            fm_step_status = self.status_dict["steps"][msg.step.index]

        if isinstance(msg, Init):
            logger.debug("Init Message Instance")
            self._delete_old_status_files()
            self._init_status_file()
            self.status_dict = self._init_step_status_dict(
                msg.timestamp, msg.run_id, msg.steps
            )

        elif isinstance(msg, Start):
            assert msg.step is not None
            if msg.success():
                logger.debug(
                    f"Forward model step {msg.step.name()} was successfully started"
                )
                self._start_status_file(msg)
                self._add_log_line(msg.step)
                fm_step_status["status"] = StepStatus.RUNNING
                fm_step_status["start_time"] = datetime_serialize(msg.timestamp)
            else:
                logger.error(f"Forward model step {msg.step.name()} FAILED to start")
                fm_step_status["status"] = StepStatus.FAILURE
                fm_step_status["error"] = msg.error_message
                fm_step_status["end_time"] = datetime_serialize(msg.timestamp)
                self._complete_status_file(msg)

        elif isinstance(msg, Exited):
            assert msg.step is not None
            fm_step_status["end_time"] = datetime_serialize(msg.timestamp)
            if msg.success():
                logger.debug(
                    f"Forward model step {msg.step.name()} exited successfully"
                )
                fm_step_status["status"] = StepStatus.SUCCESS
                self._complete_status_file(msg)
            else:
                logger.error(
                    STEP_EXIT_FAILED_STRING_TEMPLATE.format(
                        step_name=msg.step.name(),
                        exit_code=msg.exit_code,
                        error_message=msg.error_message,
                    )
                )
                fm_step_status["status"] = StepStatus.FAILURE
                fm_step_status["error"] = msg.error_message
                # A STATUS_file is not written if there is no exit_code, i.e.
                # when the step is killed due to timeout.
                if msg.exit_code:
                    self._complete_status_file(msg)
                self._dump_error_file(msg.step, msg.error_message)

        elif isinstance(msg, Running):
            fm_step_status["max_memory_usage"] = msg.memory_status.max_rss
            fm_step_status["current_memory_usage"] = msg.memory_status.rss
            fm_step_status["cpu_seconds"] = msg.memory_status.cpu_seconds
            fm_step_status["status"] = StepStatus.RUNNING
            memory_logger.info(msg.memory_status)

        elif isinstance(msg, Finish):
            logger.debug("Runner finished")
            if msg.success():
                logger.debug("Runner finished successfully")
                self.status_dict["end_time"] = datetime_serialize(msg.timestamp)
                self._dump_ok_file()
        self._dump_status_json()

    @staticmethod
    def _delete_old_status_files() -> None:
        logger.debug("Deleting old status files")
        cond_unlink(ERROR_file)
        cond_unlink(STATUS_file)
        cond_unlink(OK_file)

    @staticmethod
    def _write_status_file(msg: str) -> None:
        with open(STATUS_file, "a", encoding="utf-8") as status_file:
            status_file.write(msg)

    def _init_status_file(self) -> None:
        self._write_status_file(f"{'Current host':32}: {self.node}/{os.uname()[4]}\n")

    @staticmethod
    def _init_step_status_dict(
        start_time: datetime.datetime,
        run_id: str | None,
        steps: list["ForwardModelStep"],
    ) -> "StatusDict":
        return {
            "run_id": run_id,
            "start_time": datetime_serialize(start_time),
            "end_time": None,
            "steps": [create_step_dict(step) for step in steps],
        }

    def _start_status_file(self, msg: Start) -> None:
        assert msg.step is not None
        timestamp = msg.timestamp.strftime(TIME_FORMAT)
        step_name = msg.step.name()
        self._write_status_file(f"{step_name:32}: {timestamp} .... ")
        logger.info(
            f"Append {step_name} step starting timestamp {timestamp} to STATUS_file."
        )

    def _complete_status_file(self, msg: Start | Exited) -> None:
        assert msg.step is not None
        status: str = ""
        timestamp = msg.timestamp.strftime(TIME_FORMAT)
        if not msg.success():
            # There was no status code in the case of STARTUP_ERROR, so use
            # an arbitrary code less than -9.
            exit_code = -10 if isinstance(msg, Start) else msg.exit_code
            status = f" EXIT: {exit_code}/{msg.error_message}"
            logger.error(f"{msg.step.name()} step, {timestamp} {status}")
        self._write_status_file(f"{timestamp}  {status}\n")

    @staticmethod
    def _add_log_line(step: "ForwardModelStep") -> None:
        with open(LOG_file, "a", encoding="utf-8") as f:
            assert step.step_data["argList"] is not None
            args = " ".join(step.step_data["argList"])
            time_str = time.strftime(TIME_FORMAT, time.localtime())
            f.write(f"{time_str}  Calling: {step.step_data['executable']} {args}\n")

    @staticmethod
    def _dump_error_file(fm_step: "ForwardModelStep", error_msg: str | None) -> None:
        with open(ERROR_file, "w", encoding="utf-8") as file:
            file.write("<error>\n")
            file.write(
                f"  <time>{time.strftime(TIME_FORMAT, time.localtime())}</time>\n"
            )
            file.write(f"  <step>{fm_step.name()}</step>\n")
            file.write(f"  <reason>{error_msg}</reason>\n")
            stderr_file = None
            if fm_step.std_err:
                if os.path.exists(fm_step.std_err):
                    with open(fm_step.std_err, encoding="utf-8") as error_file_handler:
                        stderr = error_file_handler.read()
                        if stderr:
                            stderr_file = os.path.join(os.getcwd(), fm_step.std_err)
                        else:
                            stderr = f"Empty stderr from {fm_step.name()}\n"
                else:
                    stderr = f"stderr: Could not find file: {fm_step.std_err}\n"
            else:
                stderr = "stderr: Not redirected\n"

            # Escape XML characters
            stderr = (
                stderr.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&apos;")
            )

            file.write(f"  <stderr>\n{stderr}</stderr>\n")
            if stderr_file:
                file.write(f"  <stderr_file>{stderr_file}</stderr_file>\n")

            file.write("</error>\n")

    @staticmethod
    def _dump_ok_file() -> None:
        with open(OK_file, "w", encoding="utf-8") as f:
            f.write(
                f"All jobs complete {time.strftime(TIME_FORMAT, time.localtime())} \n"
            )

    def _dump_status_json(self) -> None:
        with open(STATUS_json, "wb") as fp:
            fp.write(orjson.dumps(self.status_dict, option=orjson.OPT_INDENT_2))
