import os
import sys
from ert_shared.storage.main import run_server


def terminate_on_parent_death():
    """Quit the server when the parent does a SIGABRT or is otherwise destroyed.
    This functionality has existed on Linux for a good while, but it isn't
    exposed in the Python standard library. Use ctypes to hook into the
    functionality.
    """
    if sys.platform != "linux" or "ERT_COMM_FD" not in os.environ:
        return

    from ctypes import CDLL, c_int, c_ulong
    import signal

    lib = CDLL(None)

    # from <sys/prctl.h>
    # int prctl(int option, ...)
    prctl = lib.prctl
    prctl.restype = c_int
    prctl.argtypes = (c_int, c_ulong)

    # from <linux/prctl.h>
    PR_SET_PDEATHSIG = 1

    # connect parent death signal to our SIGTERM
    prctl(PR_SET_PDEATHSIG, signal.SIGTERM)


if __name__ == "__main__":
    import ert_logging

    terminate_on_parent_death()
    run_server(debug=False)
