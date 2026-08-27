from __future__ import annotations

import contextlib
import importlib.util
import inspect
import io
import logging
import sys
import threading
import traceback
from abc import abstractmethod
from collections.abc import Iterable, Iterator
from types import MappingProxyType, ModuleType
from typing import Any, TextIO, override

from .workflow_fixtures import (
    WorkflowFixtures,
    all_hooked_workflow_fixtures,
)

logger = logging.getLogger(__name__)


class _CaptureProxy(io.TextIOBase):
    """Stands in for ``sys.stdout``/``sys.stderr`` while workflow jobs run.

    Everything written is passed on to the stream it replaced, so output still
    reaches the terminal. Needed for capturing prints from internal jobs and
    adding them to the log.
    """

    def __init__(self, stream: TextIO | None) -> None:
        super().__init__()
        self.wrapped = stream
        self._local = threading.local()
        self._users = 0

    @property
    def _buffers(self) -> list[io.StringIO]:
        buffers: list[io.StringIO] | None = getattr(self._local, "buffers", None)
        if buffers is None:
            buffers = []
            self._local.buffers = buffers
        return buffers

    @contextlib.contextmanager
    def capture(self) -> Iterator[io.StringIO]:
        """Record what the calling thread writes for the duration of the block."""
        buffer = io.StringIO()
        buffers = self._buffers
        buffers.append(buffer)
        try:
            yield buffer
        finally:
            buffers.remove(buffer)

    @override
    def write(self, s: str, /) -> int:
        for buffer in self._buffers:
            buffer.write(s)
        if self.wrapped is None:
            return len(s)
        return self.wrapped.write(s)

    @override
    def writelines(self, lines: Iterable[str], /) -> None:  # type: ignore[override]
        for line in lines:
            self.write(line)

    @override
    def flush(self) -> None:
        if self.wrapped is not None:
            self.wrapped.flush()

    @override
    def close(self) -> None:
        """Closing is ignored, as the wrapped stream outlives the capture."""

    @override
    def fileno(self) -> int:
        if self.wrapped is None:
            raise io.UnsupportedOperation("fileno")
        return self.wrapped.fileno()

    @override
    def isatty(self) -> bool:
        return self.wrapped is not None and self.wrapped.isatty()

    @override
    def writable(self) -> bool:
        return True

    @override
    def readable(self) -> bool:
        return False

    @override
    def seekable(self) -> bool:
        return False

    @property
    @override
    def encoding(self) -> str:  # type: ignore[override]
        return getattr(self.wrapped, "encoding", "utf-8")

    @property
    @override
    def errors(self) -> str | None:  # type: ignore[override]
        return getattr(self.wrapped, "errors", None)

    @property
    @override
    def newlines(self) -> Any:  # type: ignore[override]
        return getattr(self.wrapped, "newlines", None)

    @property
    def buffer(self) -> Any:
        return self.wrapped.buffer  # type: ignore[union-attr]

    @property
    def name(self) -> Any:
        return getattr(self.wrapped, "name", None)

    @property
    def line_buffering(self) -> bool:
        return getattr(self.wrapped, "line_buffering", False)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__dict__["wrapped"], name)


_capture_lock = threading.Lock()


@contextlib.contextmanager
def _capturing(stream_name: str) -> Iterator[io.StringIO]:
    # Record what the calling thread writes to ``sys.<stream_name>``
    with _capture_lock:
        stream = getattr(sys, stream_name)
        proxy = stream if isinstance(stream, _CaptureProxy) else _CaptureProxy(stream)
        proxy._users += 1
        setattr(sys, stream_name, proxy)
    try:
        with proxy.capture() as buffer:
            yield buffer
    finally:
        with _capture_lock:
            proxy._users -= 1
            if proxy._users == 0 and getattr(sys, stream_name) is proxy:
                setattr(sys, stream_name, proxy.wrapped)


class ExternalScriptError(RuntimeError):
    """Raised when an external workflow job exits with a non-zero exit code.

    Reported without a stack trace, since it would only
    show ert internals and could be confusing
    """


class ErtScript:
    """
    ErtScript is the abstract baseclass for workflow jobs and
    plugins. It provides access to the ert internals and lets
    jobs implement the "run" function which is called when
    a workflow is executed.
    """

    stop_on_fail = False

    def __init__(
        self,
    ) -> None:
        self.__is_cancelled = False
        self.__failed = False
        self._stdoutdata = ""
        self._stderrdata = ""

    @abstractmethod
    def run(self, *arg: Any, **kwarg: Any) -> Any:
        """
        This method is implemented by the workflow runners
        and executed when the workflow job is called.

        The parameters are gotten from the workflow file, e.g. a
        workflow file containing

        EXPORT_MISFIT_DATA path/to/output.hdf

        will put `path/to/output.hdf` in the first argument
        to run.
        """

    @property
    def stdoutdata(self) -> str:
        if isinstance(self._stdoutdata, bytes):
            self._stdoutdata = self._stdoutdata.decode()
        return self._stdoutdata

    @property
    def stderrdata(self) -> str:
        if isinstance(self._stderrdata, bytes):
            self._stderrdata = self._stderrdata.decode()
        return self._stderrdata

    def isCancelled(self) -> bool:
        return self.__is_cancelled

    def hasFailed(self) -> bool:
        return self.__failed

    def cancel(self) -> None:
        self.__is_cancelled = True

    def cleanup(self) -> None:
        """Override to perform cleanup after a run."""

    @property
    def requested_fixtures(self) -> set[str]:
        return {
            k
            for k in inspect.signature(self.run).parameters
            if k in all_hooked_workflow_fixtures
        }

    def initializeAndRun(
        self,
        argument_types: list[type[Any]],
        argument_values: list[str],
        fixtures: WorkflowFixtures | None = None,
    ) -> Any:
        fixtures = {} if fixtures is None else fixtures
        arguments = []
        for index, arg_value in enumerate(argument_values):
            arg_type = argument_types[index] if index < len(argument_types) else str

            if arg_value is not None:
                arguments.append(arg_type(arg_value))
            else:
                arguments.append(None)
        fixtures["workflow_args"] = arguments
        try:  # ruff: ignore[too-many-statements-in-try-clause]
            func_args = inspect.signature(self.run).parameters
            # If the user has specified *args, we skip injecting fixtures, and just
            # pass the user configured arguments
            if not any(p.kind == p.VAR_POSITIONAL for p in func_args.values()):
                try:
                    arguments = self.insert_fixtures(func_args, fixtures)
                except ValueError as e:
                    # This is here for backwards compatibility, the user does
                    # not have *argv but positional arguments. Can not be
                    # mixed with using fixtures.
                    logger.warning(
                        f"Mixture of fixtures and positional arguments, err: {e}"
                    )

            return self._run_capturing_output(arguments)
        except AttributeError as e:
            error_msg = str(e)
            if not hasattr(self, "run"):
                error_msg = "No 'run' function implemented"
            self.output_stack_trace(error=error_msg)
            logger.error(
                f"Attribute error in workflow script {self.__class__.__name__}:"
                f" {error_msg}"
            )
            return None
        except KeyboardInterrupt:
            error_msg = "Script cancelled (CTRL+C)"
            self.output_stack_trace(error=error_msg)
            logger.info(
                f"Script cancelled in workflow script {self.__class__.__name__}:"
                f" {error_msg}"
            )
            return None
        except UserWarning as uw:
            self.__failed = True
            self.output_stack_trace(error=str(uw))
            logger.warning(
                f"User warning in workflow script {self.__class__.__name__}: {uw}"
            )
            return uw.args[0]
        except ExternalScriptError as e:
            self.output_stack_trace(error=str(e))
            logger.error(f"Workflow job failed: {e!s}")
            return None
        except BaseException as e:
            full_trace = "".join(traceback.format_exception(*sys.exc_info()))
            self.output_stack_trace(f"{e!s}\n{full_trace}")
            logger.exception(
                f"Exception in workflow script {self.__class__.__name__}:"
                f" {e!s}\n{full_trace}"
            )
            return None
        finally:
            self.cleanup()

    def _run_capturing_output(self, arguments: list[Any]) -> Any:
        with _capturing("stdout") as stdout, _capturing("stderr") as stderr:
            try:
                return self.run(*arguments)
            finally:
                self._stdoutdata = self.stdoutdata + stdout.getvalue()
                self._stderrdata = self.stderrdata + stderr.getvalue()

    # Need to have unique modules in case of identical object naming in scripts
    __module_count = 0

    def insert_fixtures(
        self,
        func_args: MappingProxyType[str, inspect.Parameter],
        fixtures: WorkflowFixtures,
    ) -> list[Any]:
        arguments = []
        errors = []
        for val in func_args:
            if val in fixtures:
                arguments.append(fixtures.get(val))
            else:
                errors.append(val)
        if errors:
            raise ValueError(
                f"Plugin: {self.__class__.__name__} misconfigured, arguments: {errors} "
                f"not found in fixtures: {list(fixtures)}"
            )
        return arguments

    def output_stack_trace(self, error: str = "") -> None:
        stack_trace = error or "".join(traceback.format_exception(*sys.exc_info()))
        sys.stderr.write(
            f"The script '{self.__class__.__name__}' caused an "
            f"error while running:\n{str(stack_trace).strip()}\n"
        )

        existing_stderr = self.stderrdata
        if existing_stderr and not existing_stderr.endswith("\n"):
            existing_stderr += "\n"
        self._stderrdata = existing_stderr + error
        self.__failed = True

    @staticmethod
    def loadScriptFromFile(
        path: str,
    ) -> type[ErtScript]:
        module_name = f"ErtScriptModule_{ErtScript.__module_count}"
        ErtScript.__module_count += 1

        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None:
            raise ValueError(f"Could not find spec for {module_name}")
        module = importlib.util.module_from_spec(spec)
        if module is None:
            raise ValueError(f"Could not find {module_name} with spec {spec}")
        if spec.loader is None:
            raise ValueError(f"No loader for module {module} with spec {spec}")
        try:
            spec.loader.exec_module(module)
        except (SyntaxError, ImportError) as err:
            raise ValueError(f"ErtScript {path} contains syntax error {err}") from err
        return ErtScript.__findErtScriptImplementations(module)

    @staticmethod
    def __findErtScriptImplementations(
        module: ModuleType,
    ) -> type[ErtScript]:
        result = None
        for _, member in inspect.getmembers(
            module,
            lambda member: (
                inspect.isclass(member) and member.__module__ == module.__name__
            ),
        ):
            if ErtScript in inspect.getmro(member):
                if result is not None:
                    raise ValueError(
                        f"Module {module.__name__} contains more than one ErtScript"
                    )
                result = member

        if result is None:
            raise ValueError(f"Module {module.__name__} does not contain an ErtScript!")
        return result

    @staticmethod
    def validate(args: list[Any]) -> None:
        """
        If the workflow has problems it can validate against
        the arguments on startup. If it raises ConfigValidationError
        this will be caught and presented to the user.
        """
