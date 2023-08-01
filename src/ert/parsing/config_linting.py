from ert._c_wrappers.enkf import ErtConfig

from .config_errors import ConfigValidationError


def lint_file(file: str) -> None:
    try:
        ErtConfig.from_file(file)
        print("Found no errors")

    except ConfigValidationError as err:
        print("\n".join(m.replace("\n", " ") for m in err.get_error_messages()))
