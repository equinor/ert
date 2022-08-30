import os
from typing import TYPE_CHECKING, Any, List, Optional, Type, Union

from cwrap import BaseCClass

from ert._c_wrappers import ResPrototype
from ert._c_wrappers.config import ContentTypeEnum
from ert._c_wrappers.job_queue import (
    ErtPlugin,
    ErtScript,
    ExternalErtScript,
    FunctionErtScript,
)

if TYPE_CHECKING:
    from ert._c_wrappers.enkf import EnKFMain

    ContentTypes = Union[Type[int], Type[bool], Type[float], Type[str]]


class WorkflowJob(BaseCClass):
    TYPE_NAME = "workflow_job"
    _alloc = ResPrototype("void* workflow_job_alloc(char*, bool)", bind=False)
    _alloc_parser = ResPrototype(
        "config_parser_obj workflow_job_alloc_config( )", bind=False
    )
    _alloc_from_file = ResPrototype(
        "workflow_job_obj workflow_job_config_alloc( char* , config_parser , char*)",
        bind=False,
    )
    _free = ResPrototype("void     workflow_job_free(workflow_job)")
    _name = ResPrototype("char*    workflow_job_get_name(workflow_job)")
    _internal = ResPrototype("bool     workflow_job_internal(workflow_job)")
    _is_internal_script = ResPrototype(
        "bool   workflow_job_is_internal_script(workflow_job)"
    )
    _get_internal_script = ResPrototype(
        "char*  workflow_job_get_internal_script_path(workflow_job)"
    )
    _get_function = ResPrototype("char*  workflow_job_get_function(workflow_job)")
    _get_executable = ResPrototype("char*  workflow_job_get_executable(workflow_job)")
    _min_arg = ResPrototype("int  workflow_job_get_min_arg(workflow_job)")
    _max_arg = ResPrototype("int  workflow_job_get_max_arg(workflow_job)")
    _arg_type = ResPrototype(
        "config_content_type_enum workflow_job_iget_argtype(workflow_job, int)"
    )

    @classmethod
    def configParser(cls):
        return cls._alloc_parser()

    @classmethod
    def fromFile(cls, config_file, name=None, parser=None):
        if os.path.isfile(config_file) and os.access(config_file, os.R_OK):
            if parser is None:
                parser = cls.configParser()

            if name is None:
                name = os.path.basename(config_file)

            # NB: Observe argument reoredring.
            return cls._alloc_from_file(name, parser, config_file)
        else:
            raise IOError(f"Could not open config_file:{config_file}")

    def __init__(self, name, internal=True):
        c_ptr = self._alloc(name, internal)
        super().__init__(c_ptr)

        self.__script: Optional[ErtScript] = None
        self.__running = False

    def isInternal(self) -> bool:
        return self._internal()

    def name(self) -> str:
        return self._name()

    def minimumArgumentCount(self) -> int:
        return self._min_arg()

    def maximumArgumentCount(self) -> int:
        return self._max_arg()

    def functionName(self) -> str:
        return self._get_function()

    def executable(self) -> str:
        return self._get_executable()

    def isInternalScript(self) -> bool:
        return self._is_internal_script()

    def getInternalScriptPath(self) -> str:
        return self._get_internal_script()

    def isPlugin(self) -> bool:
        if self.isInternalScript():
            script_obj = ErtScript.loadScriptFromFile(self.getInternalScriptPath())
            return script_obj is not None and issubclass(script_obj, ErtPlugin)

        return False

    def argumentTypes(self) -> List[Optional["ContentTypes"]]:
        result: List[Optional["ContentTypes"]] = []
        for index in range(self.maximumArgumentCount()):
            t = self._arg_type(index)
            if t == ContentTypeEnum.CONFIG_BOOL:
                result.append(bool)
            elif t == ContentTypeEnum.CONFIG_FLOAT:
                result.append(float)
            elif t == ContentTypeEnum.CONFIG_INT:
                result.append(int)
            elif t == ContentTypeEnum.CONFIG_STRING:
                result.append(str)
            else:
                result.append(None)

        return result

    @property
    def execution_type(self):
        if self.isInternal() and self.isInternalScript():
            return "internal python"
        elif self.isInternal():
            return "internal C"
        return "external"

    def run(self, ert: "EnKFMain", arguments: List[str], verbose: bool = False) -> Any:
        self.__running = True

        min_arg = self.minimumArgumentCount()
        if min_arg > 0 and len(arguments) < min_arg:
            raise UserWarning(
                f"The job: {self.name()} requires at least "
                f"{min_arg} arguments, {len(arguments)} given."
            )

        max_arg = self.maximumArgumentCount()
        if 0 < max_arg < len(arguments):
            raise UserWarning(
                f"The job: {self.name()} can only have "
                f"{max_arg} arguments, {len(arguments)} given."
            )

        if self.isInternalScript():
            script_obj = ErtScript.loadScriptFromFile(self.getInternalScriptPath())
            self.__script = script_obj(ert)
            result = self.__script.initializeAndRun(
                self.argumentTypes(), arguments, verbose=verbose
            )

        elif self.isInternal() and not self.isInternalScript():
            self.__script = FunctionErtScript(
                ert,
                self.functionName(),
                self.argumentTypes(),
                argument_count=len(arguments),
            )
            result = self.__script.initializeAndRun(
                self.argumentTypes(), arguments, verbose=verbose
            )

        elif not self.isInternal():
            self.__script = ExternalErtScript(ert, self.executable())
            result = self.__script.initializeAndRun(
                self.argumentTypes(), arguments, verbose=verbose
            )

        else:
            raise UserWarning("Unknown script type!")

        self.__running = False
        return result

    def cancel(self) -> None:
        if self.__script is not None:
            self.__script.cancel()

    def isRunning(self) -> bool:
        return self.__running

    def isCancelled(self) -> bool:
        if self.__script is None:
            raise ValueError("The job must be run before calling isCancelled")
        return self.__script.isCancelled()

    def hasFailed(self) -> bool:
        if self.__script is None:
            raise ValueError("The job must be run before calling hasFailed")
        return self.__script.hasFailed()

    def free(self):
        self._free()

    def stdoutdata(self) -> str:
        if self.__script is None:
            raise ValueError("The job must be run before getting stdoutdata")
        return self.__script.stdoutdata

    def stderrdata(self) -> str:
        if self.__script is None:
            raise ValueError("The job must be run before getting stderrdata")
        return self.__script.stderrdata

    @classmethod
    def createCReference(cls, c_pointer, parent=None):
        workflow = super().createCReference(c_pointer, parent)
        workflow.__script = None
        workflow.__running = False
        return workflow

    def __ne__(self, other) -> bool:
        return not self == other

    def __eq__(self, other) -> bool:

        if self.executable() != other.executable():
            return False

        if self._is_internal_script() != other._is_internal_script():
            return False

        if (
            self._is_internal_script()
            and self._get_internal_script() != other._get_internal_script()
        ):
            return False

        if self._name() != other._name():
            return False

        if self._min_arg() != other._min_arg():
            return False

        if self._max_arg() != other._max_arg():
            return False

        return True
