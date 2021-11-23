"""
The :py:mod:`ert3.workspace` module is reponsible for persisting and handling
configuration and input data, and making those accessible to the remainder of
ERT via a high-level API. The actual representation of the workspace data should
not leak out of this module, allowing the underlying implementation to be
exchanged as needed. Moreover, the workspace API and its implemenation are
agnostic of the storage solution used by ERT.

The current implementation stores configuration and input data on disk in a
workspace directory, which is marked as a `workspace` by the presence of a
hidden :file:`.ert` directory, that may be used to store internal data that is
not to be exposed to the user. Other than that, the contents of the workspace
directory should be immutable, apart from output files generated after running
experiments.

The high-level API that exposes the configuration and input data is implemented
as a class (:py:class:`ert3.workspace.Workspace`). The persisted configuration
and input data is initialized via a separate initialization function
(:py:func:`ert3.workspace.initialize`), and can be accessed repeatedly by
creating workspace objects.

.. note:: The current file-based implementation still leaks some of the
   underlying representation: some resources in the workspace directory are
   stored in a :file:`resources` directory, and its path is exposed by the
   :py:meth:`ert3.workspace.Workspace.get_resources_dir` method.
"""

from ert3.workspace._workspace import Workspace
from ert3.workspace._workspace import initialize

__all__ = [
    "Workspace",
    "initialize",
]
