import warnings

warnings.simplefilter('always', DeprecationWarning) # see #1437
warnings.warn(
    "Importing from ert is deprecated and will be removed in a future release, use: \'from res import\' instead",
    DeprecationWarning
)

try:
    from .local import *
except ImportError:
    pass
