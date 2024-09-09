try:
    from .version import version as __version__
except ImportError:
    __version__ = "0.0.0"

import importlib.util
from pathlib import Path


def ert_share_path() -> str:
    spec = importlib.util.find_spec("ert.shared")
    assert spec, "Could not find ert.shared in import path"
    assert spec.has_location
    spec_origin = spec.origin
    assert spec_origin
    return str(Path(spec_origin).parent.parent / "resources")


from .net_utils import find_available_socket, get_machine_name

__all__ = ["__version__", "ert_share_path", "find_available_socket", "get_machine_name"]
