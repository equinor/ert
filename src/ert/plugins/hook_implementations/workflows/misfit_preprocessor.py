from typing import Any, List

from ert.config.ert_script import ErtScript
from ert.config.parsing.config_errors import ConfigValidationError


class MisfitPreprocessor(ErtScript):
    """MISFIT_PREPROCESSOR is removed, use ANALYSIS_SET_VAR OBSERVATIONS AUTO_SCALE"
    example: ANALYSIS_SET_VAR OBSERVATIONS AUTO_SCALE *  -- all observations"
    """

    def run(self, disable_parameters: str) -> None:
        raise NotImplementedError(MisfitPreprocessor.__doc__)

    @staticmethod
    def validate(args: List[Any]) -> None:
        message = MisfitPreprocessor.__doc__
        assert message is not None
        if args:
            # This means the user has configured a config file to the workflow
            # so we can assume they have customized the obs groups
            message += (
                "example: ANALYSIS_SET_VAR OBSERVATIONS AUTO_SCALE 'obs_*'  -- all observations starting with obs_"
                "Add multiple entries to set up multiple groups"
            )
        raise ConfigValidationError(message)
