"""pre-fork execution of code that is likely to cause post-fork issues on macOS
"""

import sys


def _preload_scproxy():
    """
    After the introduction of the StorageTransmitter, tests and ert3 on macOS
    started crashing. The cause of these crashes are:
    1. A change in macOS where fork-unsafe usage of core system libraries (SDK)
    would crash. See https://bugs.python.org/issue33725. fork-unsafe usage
    refers to calls made after fork() but before exec().
    2. multiprocessing's forkserver essentially does spawn()/exec() for the server
    itself, then fork() but no exec() for all instantiations of
    multiprocessing.Process. This allows for potentially fork-unsafe calls in
    the multiprocessing.Process. forkserver is used by ert3 and chosen for for
    its speed and relative robust approach.
    3. ert3 somehow manages to trigger these crashes. It's presently unclear
    exactly what in ert3 causes this, but no application, no matter how simple,
    seems to be safe from this bug.

    This module should take aim at calling all so-called +initialize methods of the
    macOS SDK before the forkserver ever calls fork(). This is only viable when code
    executed post-fork is known.
    This is likely to break down if the ert3 user introduces arbitrary code execution
    in a FunctionStep. See https://github.com/equinor/ert/issues/2045

    Ensure requests make its _scproxy [1] calls so that they don't have to be repeated
    after the fork(). StorageTransmitters will at the very least make this call, so it
    needs to be made here as well.
    [1] https://github.com/python/cpython/blob/main/Modules/_scproxy.c
    """
    import requests

    try:
        requests.get("http://localhost")
    except requests.RequestException:
        pass


if sys.platform == "darwin":
    _preload_scproxy()
