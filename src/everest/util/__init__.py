from ropt.version import version as ropt_version

try:
    from ert.shared.version import version as ert_version
except ImportError:
    ert_version = "0.0.0"


def version_info() -> str:
    return f"everest:{ert_version}, ropt:{ropt_version}, ert:{ert_version}"
