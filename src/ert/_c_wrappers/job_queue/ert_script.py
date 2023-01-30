import importlib.util
import inspect
import logging
import sys
import traceback
from typing import TYPE_CHECKING, Any, Callable, List, Type

from ert._c_wrappers.job_queue.run_status import RunStatus

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
        self.__run_status = RunStatus()

    def ert(self) -> "EnKFMain":
        logger.info(f"Accessing EnKFMain from workflow: {self.__class__.__name__}")
        return self.__ert

    @property
    def run_status(self):
        return self.__run_status

    def cancel(self):
        self.run_status.cancel()

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
            self.run_status.start()
            msg = self.run(*arguments)
            self.run_status.finish(msg)
        except AttributeError as e:
            error_msg = str(e)
            if not hasattr(self, "run"):
                error_msg = "No 'run' function implemented"
            self.output_stack_trace(error=error_msg)
        except KeyboardInterrupt:
            error_msg = "Script cancelled (CTRL+C)"
            self.output_stack_trace(error=error_msg)
        except Exception as e:
            self.output_stack_trace(str(e))
        finally:
            self.cleanup()

    # Need to have unique modules in case of identical object naming in scripts
    __module_count = 0

    def output_stack_trace(self, error: str = ""):
        stack_trace = error or "".join(traceback.format_exception(*sys.exc_info()))
        msg = (
            f"The script '{self.__class__.__name__}' caused an "
            f"error while running:\n{str(stack_trace).strip()}\n"
        )
        sys.stderr.write(msg)
        self.run_status.error(msg)

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
