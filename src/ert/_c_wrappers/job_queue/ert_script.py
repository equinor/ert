import importlib.util
import inspect
import logging
import sys
import traceback
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ert._c_wrappers.enkf import EnKFMain

logger = logging.getLogger(__name__)


class ErtScript:
    def __init__(self, ert: "EnKFMain"):
        if not hasattr(self, "run"):
            raise UserWarning(
                "ErtScript implementations must provide a method run(self, ert, ...)"
            )
        self.__verbose = False
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

    def isVerbose(self):
        return self.__verbose

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

    def initializeAndRun(self, argument_types, argument_values, verbose=False):
        """
        @type argument_types: list of type
        @type argument_values: list of string
        @type verbose: bool
        @rtype: unknown
        """
        self.__verbose = verbose
        self.__failed = False

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
            if not hasattr(self, "run"):
                self.__failed = True
                return (
                    f"Script '{self.__class__.__name__}' "
                    "has not implemented a 'run' function"
                )
            self.outputStackTrace(e)
            return None
        except KeyboardInterrupt:
            return f"Script '{self.__class__.__name__}' cancelled (CTRL+C)"
        except Exception as e:
            self.outputStackTrace(e)
            return None
        finally:
            self.cleanup()

    __module_count = (
        0  # Need to have unique modules in case of identical object naming in scripts
    )

    def outputStackTrace(self, error=None):
        stack_trace = error or "".join(traceback.format_exception(*sys.exc_info()))
        sys.stderr.write(
            f"The script '{self.__class__.__name__}' caused an "
            f"error while running:\n{str(stack_trace).strip()}\n"
        )
        self.__failed = True

    @staticmethod
    def loadScriptFromFile(path) -> Optional["ErtScript"]:
        try:
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
            spec.loader.exec_module(module)
            return ErtScript.__findErtScriptImplementations(module)
        except Exception:
            sys.stderr.write(f"The script '{path}' caused an error during load:\n")
            traceback.print_exception(sys.exc_info()[0], sys.exc_info()[1], None)
            return None

    @staticmethod
    def __findErtScriptImplementations(module) -> "ErtScript":
        result = []
        predicate = (
            # pylint: disable=unnecessary-lambda-assignment
            lambda member: inspect.isclass(member)
            and member.__module__ == module.__name__
        )
        for _, member in inspect.getmembers(module, predicate):
            if ErtScript in inspect.getmro(member):
                result.append(member)

        if len(result) != 1:
            raise UserWarning(
                "Must have (only) one implementation of ErtScript in a module!"
            )

        return result[0]
