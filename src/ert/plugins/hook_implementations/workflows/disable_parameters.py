from typing import Any, List

from ert.config.ert_script import ErtScript
from ert.config.parsing.config_errors import ConfigValidationError


class DisableParametersUpdate(ErtScript):
    """The DISABLE_PARAMETERS workflow has been moved to ert parameters.
    Add GEN_KW <name> ... UPDATE:FALSE
    """

    def run(self, disable_parameters: str) -> None:
        raise NotImplementedError(DisableParametersUpdate.__doc__)

    @staticmethod
    def validate(args: List[Any]) -> None:
        raise ConfigValidationError(
            f"DISABLE_PARAMETERS is removed, use the UPDATE:FALSE "
            f"option to the parameter instead:"
            f"example: GEN_KW {args[0]} ... UPDATE:FALSE"
        )
