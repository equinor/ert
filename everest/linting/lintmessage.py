"""Immutable container for lint error messages.

The contain the error_type (e.g. VALUE_ERROR), the key (the key in the
configuration that failed linting), and an optional error message that should
provide the user valuable debug information.

"""


class LintMessage(object):
    def __init__(self, key, error_type, msg=None):
        self._key = str(key)  # a str so we can return it in __str__
        self._error_type = error_type
        if not msg:
            msg = ""
        self._msg = msg

    @property
    def key(self):
        return self._key

    @property
    def error_type(self):
        return self._error_type

    @property
    def msg(self):
        return self._msg

    def __str__(self):
        return self.key

    def __repr__(self):
        return 'LintMessage(%s, %s, msg="%s")' % (self.key, self.error_type, self.msg)

    def hash(self):
        return hash((self.key, self.error_type, self.msg))
