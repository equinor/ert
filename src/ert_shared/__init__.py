try:
    from ert_shared.version import version as __version__
except ImportError:
    __version__ = "0.0.0"

# Other modules depend on the ert shared resources so we explicitly expose their path
from ert_shared.hook_implementations.jobs import (
    _resolve_ert_share_path as ert_share_path,
)
