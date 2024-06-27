"""
The job_queue package contains modules and classes for running
external commands.
"""

# Getting LSF to work properly is quite painful. The situation
# is a mix of build complexity and LSF specific requirements:
#
#   1. The LSF libraries are accessed from the libjob_queue.so
#      library, but observe that the dependancy on the liblsf and
#      libbat libraries is through dlopen(), i.e. runtime. This module
#      will therefore load happily without access to the lsf libraries.
#
#      If you at a later stage create a lsf driver the runtime
#      environment must be able to locate the liblsf.so, libbat.so and
#      libnsl.so shared libraries, either through LD_LIBRARY_PATH or
#      other means.
#
#   2. To actually use LSF you need a whole list of environment
#      variables to be set: LSF_BINDIR , LSF_LIBDIR , XLDF_UIDDIR ,
#      LSF_SERVERDIR, LSF_ENVDIR - this is an LSF requirement and not
#      related to ERT or the Python bindings. The normal way to
#      achieve this is by sourcing a shell script.
#
#      If the environment variable LSF_HOME is set we set the
#      remaining LSF variables according to:
#
#           LSF_BINDIR    = $LSF_HOME/bin
#           LSF_LIBDIR    = $LSF_HOME/lib
#           XLSF_UIDDIR   = $LSF_HOME/lib/uid
#           LSF_SERVERDIR = $LSF_HOME/etc
#           LSF_ENVDIR    = $LSF_HOME/conf
#           PATH          = $PATH:$LSF_BINDIR
#
#      Observe that none of these variables are modified if they
#      already have a value, furthermore it should be observed that
#      the use of an LSF_HOME variable is something invented with ERT,
#      and not standard LSF approach.

import ctypes
import os
import os.path
import warnings
from typing import Any

from cwrap import Prototype  # noqa


def setenv(var: str, value: str) -> None:
    if not os.getenv(var):
        os.environ[var] = value


# Set up the full LSF environment - based onf LSF_HOME
LSF_HOME = os.getenv("LSF_HOME")
if LSF_HOME:
    setenv("LSF_BINDIR", f"{LSF_HOME}/bin")
    setenv("LSF_LIBDIR", f"{LSF_HOME}/lib")
    setenv("XLSF_UIDDIR", f"{LSF_HOME}/lib/uid")
    setenv("LSF_SERVERDIR", f"{LSF_HOME}/etc")
    setenv("LSF_ENVDIR", f"{LSF_HOME}/conf")  # This is wrong: Equinor: /prog/LSF/conf


warnings.filterwarnings(action="always", category=DeprecationWarning, module=r"res|ert")


def _load_lib() -> Any:
    import ert._clib  # noqa: PLC0415

    lib = ctypes.CDLL(ert._clib.__file__)

    return lib


class ResPrototype(Prototype):  # type: ignore
    lib = _load_lib()

    def __init__(self, prototype: str, bind: bool = True) -> None:
        super().__init__(ResPrototype.lib, prototype, bind=bind)


from .driver import Driver  # noqa
from .job_queue_node import JobQueueNode  # noqa
from .job_status import JobStatus  # noqa
from .queue import JobQueue  # noqa
from .submit_status import SubmitStatus  # noqa
from .thread_status import ThreadStatus  # noqa

__all__ = [
    "Driver",
    "JobQueue",
    "JobQueueNode",
    "JobStatus",
    "SubmitStatus",
    "ThreadStatus",
]
