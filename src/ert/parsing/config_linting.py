from ert._c_wrappers.enkf import ErtConfig
from ert.parsing import ConfigValidationError


def lint_file(file: str, is_for_problem_matcher=False):
    try:
        ErtConfig.from_file(file, use_new_parser=True)
        print("Found no errors")

    except ConfigValidationError as err:
        if is_for_problem_matcher:
            print(err.get_cli_message_for_problem_matcher())
        else:
            print(err.get_cli_message())
