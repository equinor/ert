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


import os

from cwrap import Prototype

import ert._c_wrappers  # noqa


def setenv(var, value):
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

from .driver import Driver, LocalDriver, LSFDriver, QueueDriverEnum  # noqa
from .ert_plugin import CancelPluginException, ErtPlugin  # noqa
from .ert_script import ErtScript  # noqa
from .ext_job import ExtJob, ExtJobInvalidArgsException  # noqa
from .external_ert_script import ExternalErtScript  # noqa
from .function_ert_script import FunctionErtScript  # noqa
from .job import Job  # noqa
from .job_queue_manager import JobQueueManager  # noqa
from .job_queue_node import JobQueueNode  # noqa
from .job_status_type_enum import JobStatusType  # noqa
from .job_submit_status_type_enum import JobSubmitStatusType  # noqa
from .queue import JobQueue  # noqa
from .run_status_type_enum import RunStatusType  # noqa
from .thread_status_type_enum import ThreadStatus  # noqa
from .workflow import Workflow  # noqa
from .workflow_job import WorkflowJob  # noqa
from .workflow_runner import WorkflowRunner  # noqa

__all__ = [
    "Prototype",
    "JobStatusType",
    "RunStatusType",
    "ThreadStatus",
    "JobSubmitStatusType",
    "Job",
    "Driver",
    "QueueDriverEnum",
    "JobQueueNode",
    "JobQueue",
    "JobQueueManager",
    "QueueDriverEnum",
    "Driver",
    "LSFDriver",
    "LocalDriver",
    "ExtJob",
    "ExtJobInvalidArgsException",
    "ErtScript",
    "ErtPlugin",
    "CancelPluginException",
    "FunctionErtScript",
    "ExternalErtScript",
    "WorkflowJob",
    "Workflow",
    "WorkflowRunner",
]
