#!/usr/bin/env python


class Validation(object):
    """Validation is a wrapper around bool that also has a .msg attribute.

    Usually if the object is True, the message will be the empty string.

    If the object is False, the message should be an explanation of why a
    validation failed.

    """

    def __init__(self, value, msg=None, input=None):
        self.value = value
        if not isinstance(msg, str):
            msg = ""
        self._msg = msg
        self._input = str(input)

    def __nonzero__(self):
        return self.value is True

    def __bool__(self):
        return self.value is True

    @property
    def msg(self):
        return self._msg + " on input " + self._input

    @classmethod
    def true(cls):
        return Validation(True)

    @classmethod
    def false(cls, msg="", input=None):
        return Validation(False, msg=msg, input=input)

    def __repr__(self):
        if self:
            return "Validation(True)"
        return "Validation(False, msg=%s)" % self._msg


def validator(msg):
    """Validator decorator wraps return value in a message container.

    Usage:

        @validator('assert len(x) <= 8')
        def validate_size(x):
            return len(x) <= 8

    Now, if `validate_size` returns a falsy value `ret`, we will have
        `ret.msg = 'assert len(x) <= 8 for x=obj'`
    for obj being `str(x)`.
    """

    def real_decorator(function):
        def wrapper(*args, **kwargs):
            res = function(*args, **kwargs)
            if res is True:
                return Validation.true()
            return Validation.false(msg=msg, input=str(*args))

        return wrapper

    return real_decorator
