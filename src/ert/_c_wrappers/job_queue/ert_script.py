import importlib.util
import inspect
import logging
import sys
import traceback
from typing import TYPE_CHECKING, Any, Callable, List, Type

if TYPE_CHECKING:
    from ert._c_wrappers.enkf import EnKFMain

logger = logging.getLogger(__name__)


class ErtScript:
    def __init__(self, ert: "EnKFMain"):
        if not hasattr(self, "run"):
            raise UserWarning(
                "ErtScript implementations must provide a method run(self, ert, ...)"
            )
        self.__ert = ert

        self.__is_cancelled = False
        self.__failed = False
        self._stdoutdata = ""
        self._stderrdata = ""

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

    def isCancelled(self) -> bool:
        return self.__is_cancelled

    def hasFailed(self) -> bool:
        return self.__failed

    def cancel(self):
        self.__is_cancelled = True

    def cleanup(self):
        """
        Override to perform cleanup after a run.
        """
        pass

    def initializeAndRun(
        self,
        argument_types: List[Type[Any]],
        argument_values: List[str],
    ):
        arguments = []
        for index, arg_value in enumerate(argument_values):
            if index < len(argument_types):
                arg_type = argument_types[index]
            else:
                arg_type = str

            if arg_value is not None:
                arguments.append(arg_type(arg_value))
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
            self.output_stack_trace(str(e))
            return None
        finally:
            self.cleanup()

    # Need to have unique modules in case of identical object naming in scripts
    __module_count = 0

    def output_stack_trace(self, error: str = ""):
        stack_trace = error or "".join(traceback.format_exception(*sys.exc_info()))
        sys.stderr.write(
            f"The script '{self.__class__.__name__}' caused an "
            f"error while running:\n{str(stack_trace).strip()}\n"
        )
        self.__failed = True

    @staticmethod
    def loadScriptFromFile(path) -> Callable[["EnKFMain"], "ErtScript"]:
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
    def __findErtScriptImplementations(module) -> Callable[["EnKFMain"], "ErtScript"]:
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
