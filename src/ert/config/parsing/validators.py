from ert.config.parameter_config import ParameterConfig, has_updatable_parameters
from ert.config.parsing.config_errors import ConfigValidationError


def validate_has_updatable_parameter(parameter_configs: list[ParameterConfig]) -> None:
    if not parameter_configs:
        raise ConfigValidationError(
            "No parameters to update as no GEN_KW, FIELD or SURFACE "
            "parameters are configured!"
        )
    if not has_updatable_parameters(parameter_configs):
        raise ConfigValidationError(
            "No parameters to update as all parameters were set to update:false!"
        )
