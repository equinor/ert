import contextlib
import importlib.util
import os
import sys
from io import StringIO

import pytest


def skipif_no_everviz(function):
    """Decorator to skip a test if everviz is not available"""
    spec = importlib.util.find_spec("everviz")
    not_found = spec is None
    return pytest.mark.skipif(not_found, reason="everviz not found")(function)


def skipif_no_everest_models(function):
    """Decorator to skip a test if everest-models is not available"""
    spec = importlib.util.find_spec("everest_models")
    not_found = spec is None
    return pytest.mark.skipif(not_found, reason="everest-models not found")(function)


def relpath(*path):
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), *path)


@contextlib.contextmanager
def capture_streams():
    """Context that allows capturing text sent to stdout and stderr

    Use as follow:
    with capture_streams() as (out, err):
        foo()
    assert( 'output of foo' in out.getvalue())
    """
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield new_out, new_err
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def satisfy(predicate):
    """Return a class that equals to an obj if predicate(obj) is True

    This method is expected to be used with `assert_called_with()` on mocks.
    An example can be found in `test_everest_entry.test_everest_run`
    Inspired by
    https://stackoverflow.com/questions/21611559/assert-that-a-method-was-called-with-one-argument-out-of-several
    """

    class _PredicateChecker:
        def __eq__(self, obj) -> bool:
            return predicate(obj)

    return _PredicateChecker()


def satisfy_type(the_type):
    """Specialization of satisfy for checking object type"""
    return satisfy(lambda obj: isinstance(obj, the_type))


def satisfy_callable():
    """Specialization of satisfy for checking that object is callable"""
    return satisfy(callable)


class MockParser:
    """
    Small class that contains the necessary functions in order to test custom
    validation functions used with the argparse module
    """

    def __init__(self) -> None:
        self.error_msg = None

    def get_error(self):
        return self.error_msg

    def error(self, value=None):
        self.error_msg = value
