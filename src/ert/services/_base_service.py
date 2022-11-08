import io
import json
import os
import signal
import sys
import threading
from logging import Logger, getLogger
from pathlib import Path
from select import PIPE_BUF, select
from subprocess import Popen, TimeoutExpired
from time import sleep
from types import FrameType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Type,
    TypeVar,
    Union,
)

if TYPE_CHECKING:
    from inspect import Traceback

T = TypeVar("T", bound="BaseService")
ConnInfo = Union[Mapping[str, Any], Exception, None]


SERVICE_CONF_PATHS: Set[str] = set()


def cleanup_service_files(signum: int, frame: Optional[FrameType]) -> None:
    for file_path in SERVICE_CONF_PATHS:
        file = Path(file_path)
        if file.exists():
            file.unlink()
    raise OSError(f"Signal {signum} received.")


if threading.current_thread() is threading.main_thread():
    signal.signal(signal.SIGTERM, cleanup_service_files)
    signal.signal(signal.SIGINT, cleanup_service_files)


def local_exec_args(script_args: Union[str, List[str]]) -> List[str]:
    """
    Convenience function that returns the exec_args for executing a Python
    script in the directory of '_base_service.py'.

    This is done instead of using 'python -m [module path]' due to the '-m' flag
    adding the user's current working directory to sys.path. Executing a Python
    script by itself will add the directory of the script rather than the
    current working directory, thus we avoid accidentally importing user's
    directories that just happen to have the same names as the ones we use.
    """
    if isinstance(script_args, str):
        script = script_args
        rest: List[str] = []
    else:
        script = script_args[0]
        rest = script_args[1:]
    script = f"_{script}_main.py"
    return [sys.executable, str(Path(__file__).parent / script), *rest]


class _Context(Generic[T]):
    def __init__(self, service: T) -> None:
        self._service = service

    def __enter__(self) -> T:
        return self._service

    def __exit__(
        self,
        exc_type: Type[BaseException],
        exc_value: BaseException,
        traceback: "Traceback",
    ) -> bool:
        self._service.shutdown()
        return exc_type is None


class _Proc(threading.Thread):
    def __init__(
        self,
        service_name: str,
        exec_args: Sequence[str],
        timeout: int,
        set_conn_info: Callable[[ConnInfo], None],
        project: Path,
    ):
        super().__init__()

        self._shutdown = threading.Event()

        self._service_name = service_name
        self._exec_args = exec_args
        self._timeout = timeout
        self._set_conn_info = set_conn_info
        self._service_config_path = project / f"{self._service_name}_server.json"

        fd_read, fd_write = os.pipe()
        self._comm_pipe = os.fdopen(fd_read)

        env = os.environ.copy()
        env["ERT_COMM_FD"] = str(fd_write)

        SERVICE_CONF_PATHS.add(str(self._service_config_path))

        # pylint: disable=consider-using-with
        # The process is waited for in _do_shutdown()
        self._childproc = Popen(
            self._exec_args,
            pass_fds=(fd_write,),
            env=env,
            close_fds=True,
        )
        os.close(fd_write)

    def run(self) -> None:
        comm = self._read_conn_info(self._childproc)

        if comm is None:
            self._set_conn_info(TimeoutError())
            return  # _read_conn_info() has already cleaned up in this case

        conn_info: ConnInfo = None
        try:
            conn_info = json.loads(comm)
        except json.JSONDecodeError:
            conn_info = ServerBootFail()
        except Exception as exc:
            conn_info = exc

        try:
            self._set_conn_info(conn_info)

            while True:
                if self._childproc.poll() is not None:
                    break
                if self._shutdown.wait(1):
                    self._do_shutdown()
                    break
        finally:
            self._ensure_delete_conn_info()

    def shutdown(self) -> int:
        """Shutdown the server."""
        self._shutdown.set()
        self.join()
        return self._childproc.returncode

    def _read_conn_info(self, proc: Popen) -> Optional[str]:
        comm_buf = io.StringIO()
        first_iter = True
        while first_iter or proc.poll() is None:
            first_iter = False
            ready = select([self._comm_pipe], [], [], self._timeout)

            # Timeout reached, exit with a failure
            if ready == ([], [], []):
                self._do_shutdown()
                self._ensure_delete_conn_info()
                return None

            x = self._comm_pipe.read(PIPE_BUF)
            if x == "":  # EOF
                break
            comm_buf.write(x)
        return comm_buf.getvalue()

    def _do_shutdown(self) -> None:
        if self._childproc is None:
            return
        try:
            self._childproc.terminate()
            self._childproc.wait(10)  # Give it 10s to shut down cleanly..
        except TimeoutExpired:
            try:
                self._childproc.kill()  # ... then kick it harder...
                self._childproc.wait(self._timeout)  # ... and wait again
            except TimeoutExpired:
                self.logger.error(
                    f"waiting for child-process exceeded timeout {self._timeout}s"
                )

    def _ensure_delete_conn_info(self) -> None:
        """
        Ensure that the JSON connection information file is deleted
        """
        if self._service_config_path.exists():
            self._service_config_path.unlink()

    @property
    def logger(self) -> Logger:
        return getLogger(f"ert.shared.{self._service_name}")


class ServerBootFail(RuntimeError):
    pass


class BaseService:
    """
    BaseService provides a block-only-when-needed mechanism for starting and
    maintaining services as subprocesses.

    This is achieved by using a POSIX communication pipe, over which the service
    can communicate that it has started. The contents of the communication is
    also written to a file inside of the ERT storage directory.

    The service itself can implement the other side of the pipe as such::

        import os

        # ... perform initialisation ...

        # BaseService provides this environment variable with the pipe's FD
        comm_fd = os.environ["ERT_COMM_FD"]

        # Open the pipe with Python's IO classes for ease of use
        with os.fdopen(comm_fd, "wb") as comm:
            # Write JSON over the pipe, which will be interpreted by a subclass
            # of BaseService on ERT's side
            comm.write('{"some": "json"}')

        # The pipe is flushed and closed here. This tells BaseService that
        # initialisation is finished and it will try to read the JSON data.
    """

    _instance: Optional["BaseService"] = None

    def __init__(
        self,
        exec_args: Sequence[str],
        timeout: int = 120,
        conn_info: ConnInfo = None,
        project: Optional[str] = None,
    ):
        self._exec_args = exec_args
        self._timeout = timeout

        self._proc: Optional[_Proc] = None
        self._conn_info: ConnInfo = conn_info
        self._conn_info_event = threading.Event()
        self._project = Path(project) if project is not None else Path.cwd()

        # Flag that we have connection information
        if self._conn_info:
            self._conn_info_event.set()
        else:
            self._proc = _Proc(
                self.service_name, exec_args, timeout, self.set_conn_info, self._project
            )

    @classmethod
    def start_server(cls: Type[T], *args: Any, **kwargs: Any) -> _Context[T]:
        if cls._instance is not None:
            raise RuntimeError("Server already running")
        cls._instance = obj = cls(*args, **kwargs)
        if obj._proc is not None:
            obj._proc.start()
        return _Context(obj)

    @classmethod
    def connect(
        cls: Type[T],
        *,
        project: Optional[os.PathLike] = None,
        timeout: Optional[int] = None,
    ) -> T:
        if cls._instance is not None:
            cls._instance.wait_until_ready()
            assert isinstance(cls._instance, cls)
            return cls._instance

        path = Path(project) if project is not None else Path.cwd()
        name = f"{cls.service_name}_server.json"

        # Note: If the caller actually pass None, we override that here...
        if timeout is None:
            timeout = 120
        t = -1
        while t < timeout:
            if (path / name).exists():
                with (path / name).open() as f:
                    return cls([], conn_info=json.load(f), project=str(path))

            sleep(1)
            t += 1

        raise TimeoutError("Server not started")

    @classmethod
    def connect_or_start_server(cls: Type[T], *args: Any, **kwargs: Any) -> _Context[T]:
        try:
            # Note that timeout==0 will bypass the loop in connect() and force
            # TimeoutError if there is no known existing instance
            return _Context(cls.connect(timeout=0, project=kwargs.get("project")))
        except TimeoutError:
            # Server is not running. Start a new one
            pass
        return cls.start_server(*args, **kwargs)

    def wait_until_ready(self, timeout: Optional[int] = None) -> bool:
        if timeout is None:
            timeout = self._timeout

        if self._conn_info_event.wait(timeout):
            return not (
                self._conn_info is None or isinstance(self._conn_info, Exception)
            )
        if isinstance(self._conn_info, TimeoutError):
            self.logger.critical(f"startup exceeded defined timeout {timeout}s")
        return False  # Timeout reached

    def wait(self) -> None:
        if self._proc is not None:
            self._proc.join()

    def set_conn_info(self, info: ConnInfo) -> None:
        if self._conn_info is not None:
            raise ValueError("Connection information already set")
        if info is None:
            raise ValueError
        self._conn_info = info

        if self._project is not None:
            path = f"{self._project}/{self.service_name}_server.json"
        else:
            path = f"{self.service_name}_server.json"

        if isinstance(info, Mapping):
            with open(path, "w") as f:
                json.dump(info, f, indent=4)

        self._conn_info_event.set()

    def fetch_conn_info(self) -> Mapping[str, Any]:
        is_ready = self.wait_until_ready(self._timeout)
        if isinstance(self._conn_info, Exception):
            raise self._conn_info
        if not is_ready:
            raise TimeoutError()
        if self._conn_info is None:
            raise ValueError("conn_info is None")
        return self._conn_info

    def shutdown(self) -> int:
        """Shutdown the server."""
        if self._proc is None:
            return -1
        self.__class__._instance = None
        proc, self._proc = self._proc, None
        return proc.shutdown()

    @property
    def service_name(self) -> str:
        """
        Subclass should return the name of the service, eg 'storage' for ERT Storage.
        Used for identifying the server information JSON file.
        """
        raise NotImplementedError

    @property
    def logger(self) -> Logger:
        return getLogger(f"ert.shared.{self.service_name}")

    @property
    def _service_file(self) -> str:
        return f"{self.service_name}_server.json"
