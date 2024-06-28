from __future__ import annotations

import importlib.util
import inspect
import logging
import sys
import traceback
import warnings
from abc import abstractmethod
from types import MappingProxyType, ModuleType
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type, Union

from typing_extensions import deprecated

if TYPE_CHECKING:
    from ert.config import ErtConfig
    from ert.storage import Ensemble, Storage

    Fixtures = Union[ErtConfig, Ensemble, Storage]
logger = logging.getLogger(__name__)


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

        # Deprecated:
        self._ert = None
        self._ensemble = None
        self._storage = None

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

    @deprecated("Use fixtures to the run function instead")
    def ert(self) -> Optional[ErtConfig]:
        logger.info(f"Accessing EnKFMain from workflow: {self.__class__.__name__}")
        return self._ert

    @property
    def ensemble(self) -> Optional[Ensemble]:
        warnings.warn(
            "The ensemble property is deprecated, use the fixture to the run function instead",
            DeprecationWarning,
            stacklevel=1,
        )
        logger.info(f"Accessing ensemble from workflow: {self.__class__.__name__}")
        return self._ensemble

    @property
    def storage(self) -> Optional[Storage]:
        warnings.warn(
            "The storage property is deprecated, use the fixture to the run function instead",
            DeprecationWarning,
            stacklevel=1,
        )
        logger.info(f"Accessing storage from workflow: {self.__class__.__name__}")
        return self._storage

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

    def initializeAndRun(
        self,
        argument_types: List[Type[Any]],
        argument_values: List[str],
        fixtures: Optional[Dict[str, Any]] = None,
    ) -> Any:
        fixtures = {} if fixtures is None else fixtures
        arguments = []
        for index, arg_value in enumerate(argument_values):
            arg_type = argument_types[index] if index < len(argument_types) else str

            if arg_value is not None:
                arguments.append(arg_type(arg_value))  # type: ignore
            else:
                arguments.append(None)
        fixtures["workflow_args"] = arguments
        try:
            func_args = inspect.signature(self.run).parameters
            # If the user has specified *args, we skip injecting fixtures, and just
            # pass the user configured arguments
            if not any(p.kind == p.VAR_POSITIONAL for p in func_args.values()):
                try:
                    arguments = self.insert_fixtures(func_args, fixtures)
                except ValueError as e:
                    # This is here for backwards compatibility, the user does not have *argv
                    # but positional arguments. Can not be mixed with using fixtures.
                    logger.warning(
                        f"Mixture of fixtures and positional arguments, err: {e}"
                    )
            # Part of deprecation
            self._ert = fixtures.get("ert_config")
            self._ensemble = fixtures.get("ensemble")
            self._storage = fixtures.get("storage")
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
        except UserWarning as uw:
            self.__failed = True
            return uw.args[0]
        except Exception as e:
            full_trace = "".join(traceback.format_exception(*sys.exc_info()))
            self.output_stack_trace(f"{str(e)}\n{full_trace}")
            return None
        finally:
            self.cleanup()

    # Need to have unique modules in case of identical object naming in scripts
    __module_count = 0

    def insert_fixtures(
        self,
        func_args: MappingProxyType[str, inspect.Parameter],
        fixtures: Dict[str, Fixtures],
    ) -> List[Any]:
        arguments = []
        errors = []
        for val in func_args:
            if val in fixtures:
                arguments.append(fixtures[val])
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

        self._stderrdata = error
        self.__failed = True

    @staticmethod
    def loadScriptFromFile(
        path: str,
    ) -> Callable[[], "ErtScript"]:
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
    ) -> Callable[[], "ErtScript"]:
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

    @staticmethod
    def validate(args: List[Any]) -> None:
        """
        If the workflow has problems it can validate against
        the arguments on startup. If it raises ConfigValidationError
        this will be caught and presented to the user.
        """
