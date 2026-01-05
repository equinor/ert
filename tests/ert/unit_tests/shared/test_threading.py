import threading

import pytest

from _ert.threading import ErtThread, ErtThreadError


def watch_out_im_gonna_throw():
    raise ValueError("I threw")


def test_exception_can_be_caught():
    """When a ErtThread is created and raises an exception, ensure that we are
    able to catch it and that it contains a traceback.
    """

    assert threading.current_thread() is threading.main_thread()

    # Using ErtThread throws an exception on the main thread so that pytest can
    # catch it.
    thread = ErtThread(target=watch_out_im_gonna_throw)
    thread.start()
    with pytest.raises(ErtThreadError, match="I threw") as e:
        thread.join()

    assert "Traceback (most recent call last):" in str(e)
    assert " line 9, in watch_out_im_gonna_throw" in str(e)


def test_exception_are_logged_with_traceback(caplog):
    """When a ErtThread is created and raises an exception, ensure that we are
    the exception is logged with a traceback.
    """

    assert threading.current_thread() is threading.main_thread()
    # Using ErtThread throws an exception on the main thread so that pytest can
    # catch it.
    thread = ErtThread(target=watch_out_im_gonna_throw)
    thread.start()
    with pytest.raises(ErtThreadError):
        thread.join()
    assert "Traceback (most recent call last):" in caplog.messages[0]
    assert " line 9, in watch_out_im_gonna_throw" in caplog.messages[0]
