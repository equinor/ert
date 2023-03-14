from ert._c_wrappers.config import ConfigValidationError
from ert._c_wrappers.enkf.ert_config import ErtConfig
from ert._c_wrappers.enkf.ert_config_lint_header import ErtConfigLinter


def lint_file(file: str) -> ErtConfigLinter:
    try:
        ErtConfig.from_file(user_config_file=file, use_new_parser=True)
        return ErtConfigLinter()
    except ConfigValidationError as e:
        return ErtConfigLinter.from_config_validation_error(e)
