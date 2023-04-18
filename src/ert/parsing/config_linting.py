from ert._c_wrappers.enkf import ErtConfig
from ert.parsing import ConfigValidationError


def lint_file(file: str):
    try:
        ErtConfig.from_file(file)
        print("Found no errors")

    except ConfigValidationError as err:
        print(f"Found {len(err.errors)} errors")
