from ert._c_wrappers.enkf import ErtConfig

from .config_errors import ConfigValidationError


def lint_file(file: str):
    try:
        ErtConfig.from_file(file, use_new_parser=True)
        print("Found no errors")

    except ConfigValidationError as err:
        print("\n".join(m.replace("\n", " ") for m in err.get_error_messages()))
