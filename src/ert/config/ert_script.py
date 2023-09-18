from __future__ import annotations

import importlib.util
import inspect
import logging
import sys
import traceback
from abc import abstractmethod
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Type

if TYPE_CHECKING:
    from ert.enkf_main import EnKFMain
    from ert.storage import EnsembleAccessor, StorageAccessor

logger = logging.getLogger(__name__)


class ErtScript:
    stop_on_fail = False

    def __init__(
        self,
        ert: EnKFMain,
        storage: StorageAccessor,
        ensemble: Optional[EnsembleAccessor] = None,
    ) -> None:
        self.__ert = ert
        self.__storage = storage
        self.__ensemble = ensemble

        self.__is_cancelled = False
        self.__failed = False
        self._stdoutdata = ""
        self._stderrdata = ""

    @abstractmethod
    def run(self, *arg: Any, **kwarg: Any) -> Any:
        pass

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

    def ert(self) -> "EnKFMain":
        logger.info(f"Accessing EnKFMain from workflow: {self.__class__.__name__}")
        return self.__ert

    @property
    def storage(self) -> StorageAccessor:
        return self.__storage

    @property
    def ensemble(self) -> Optional[EnsembleAccessor]:
        return self.__ensemble

    @ensemble.setter
    def ensemble(self, ensemble: EnsembleAccessor) -> None:
        self.__ensemble = ensemble

    def isCancelled(self) -> bool:
        return self.__is_cancelled

    def hasFailed(self) -> bool:
        return self.__failed

    def cancel(self) -> None:
        self.__is_cancelled = True

    def cleanup(self) -> None:
        """
        Override to perform cleanup after a run.
        """
        pass

    def initializeAndRun(
        self,
        argument_types: List[Type[Any]],
        argument_values: List[str],
    ) -> Any:
        arguments = []
        for index, arg_value in enumerate(argument_values):
            arg_type = argument_types[index] if index < len(argument_types) else str

            if arg_value is not None:
                arguments.append(arg_type(arg_value))  # type: ignore
            else:
                arguments.append(None)

        try:
            return self.run(*arguments)
        except AttributeError as e:
            error_msg = str(e)
            if not hasattr(self, "run"):
                error_msg = "No 'run' function implemented"
            self.output_stack_trace(error=error_msg)
            return None
        except KeyboardInterrupt:
            error_msg = "Script cancelled (CTRL+C)"
            self.output_stack_trace(error=error_msg)
            return None
        except Exception as e:
            full_trace = "".join(traceback.format_exception(*sys.exc_info()))
            self.output_stack_trace(f"{str(e)}\n{full_trace}")
            return None
        finally:
            self.cleanup()

    # Need to have unique modules in case of identical object naming in scripts
    __module_count = 0

    def output_stack_trace(self, error: str = "") -> None:
        stack_trace = error or "".join(traceback.format_exception(*sys.exc_info()))
        sys.stderr.write(
            f"The script '{self.__class__.__name__}' caused an "
            f"error while running:\n{str(stack_trace).strip()}\n"
        )

        self._stderrdata = error
        self.__failed = True

    @staticmethod
    def loadScriptFromFile(
        path: str,
    ) -> Callable[["EnKFMain", "StorageAccessor"], "ErtScript"]:
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
    ) -> Callable[["EnKFMain", "StorageAccessor"], "ErtScript"]:
        result = []
        for _, member in inspect.getmembers(
            module,
            lambda member: inspect.isclass(member)
            and member.__module__ == module.__name__,
        ):
            if ErtScript in inspect.getmro(member):
                result.append(member)

        if len(result) == 0:
            raise ValueError(f"Module {module.__name__} does not contain an ErtScript!")
        if len(result) > 1:
            raise ValueError(
                f"Module {module.__name__} contains more than one ErtScript"
            )

        return result[0]
