import contextlib
import logging
import os
import tempfile
import shutil

import decorator
import time

"""
Swiped from
https://github.com/equinor/everest/blob/master/tests/utils/__init__.py
"""


def tmpdir(path=None, teardown=True):
    """ Decorator based on the  `tmp` context """

    def real_decorator(function):
        def wrapper(function, *args, **kwargs):
            with tmp(path, teardown=teardown):
                return function(*args, **kwargs)

        return decorator.decorator(wrapper, function)

    return real_decorator


@contextlib.contextmanager
def tmp(path=None, teardown=True):
    """Create and go into tmp directory, returns the path.
    This function creates a temporary directory and enters that directory.  The
    returned object is the path to the created directory.
    If @path is not specified, we create an empty directory, otherwise, it must
    be a path to an existing directory.  In that case, the directory will be
    copied into the temporary directory.
    If @teardown is True (defaults to True), the directory is (attempted)
    deleted after context, otherwise it is kept as is.
    """
    cwd = os.getcwd()
    fname = tempfile.NamedTemporaryFile().name

    if path:
        if not os.path.isdir(path):
            logging.debug("tmp:raise no such path")
            raise IOError("No such directory: %s" % path)
        shutil.copytree(path, fname)
    else:
        # no path to copy, create empty dir
        os.mkdir(fname)

    os.chdir(fname)

    yield fname  # give control to caller scope

    os.chdir(cwd)

    if teardown:
        try:
            shutil.rmtree(fname)
        except OSError as oserr:
            logging.debug("tmp:rmtree failed %s (%s)" % (fname, oserr))
            shutil.rmtree(fname, ignore_errors=True)


def wait_until(func, interval=0.5, timeout=30):
    """Expects 'func' to raise an AssertionError to indicate failure.
    Repeatedly calls 'func' until it does not throw an AssertionError.
    Waits 'interval' seconds before each invocation. If 'timeout' is
    reached, will raise the AssertionError.

    Example of how to wait for a file to be created:

    wait_until(lambda: assertFileExists("/some/file"))"""
    t = 0
    while True:
        time.sleep(interval)
        t += interval
        try:
            func()
            return
        except AssertionError:
            if t >= timeout:
                raise AssertionError(
                    "Timeout reached in wait_until (function {%s}, timeout {%d}) when waiting for assertion.".format(
                        func.__name__, timeout
                    )
                )
