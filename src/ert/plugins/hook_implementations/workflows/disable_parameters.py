from typing import Any

from ert import ErtScript
from ert.config import ConfigValidationError


class DisableParametersUpdate(ErtScript):
    """The DISABLE_PARAMETERS workflow has been deprecated and removed.

    Add

        GEN_KW <name> ... UPDATE:FALSE

    to the configuration instead.
    """

    def run(self, disable_parameters: str) -> None:
        raise NotImplementedError(DisableParametersUpdate.__doc__)

    @staticmethod
    def validate(args: list[Any]) -> None:
        raise ConfigValidationError(
            f"DISABLE_PARAMETERS is removed, use the UPDATE:FALSE "
            f"option to the parameter instead:"
            f"example: GEN_KW {args[0]} ... UPDATE:FALSE"
        )
