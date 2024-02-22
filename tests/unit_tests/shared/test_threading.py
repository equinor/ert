import threading

import pytest

from ert.shared.threading import ErtThread, ErtThreadError


def test_exception_can_be_caught():
    """When a ErtThread is created and raises an exception, ensure that we are
    able to catch it

    """

    def watch_out_im_gonna_throw():
        raise ValueError("I threw")

    assert threading.current_thread() is threading.main_thread()

    # Using ErtThread throws an exception on the main thread so that pytest can
    # catch it.
    with pytest.raises(ErtThreadError, match="I threw"):
        thread = ErtThread(target=watch_out_im_gonna_throw)
        thread.start()
        thread.join()
