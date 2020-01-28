import warnings

warnings.filterwarnings(
    action='always',
    category=DeprecationWarning,
    module=r'ecl|ert|res'
)

warnings.warn(
    "Importing from ert.ecl, ecl.ecl or ert is deprecated and will not be available in python3." \
    " For eclipse functionality use \'from ecl import\', for ert workflow tooling use \'from res import\'.",
    DeprecationWarning
)

try:
    from .local import *
except ImportError:
    pass
