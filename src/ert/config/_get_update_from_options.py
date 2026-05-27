from ert.config.parameter_config import LocalizationType
from ert.config.parsing.config_errors import ConfigValidationError


def get_update_from_options(options: dict[str, str]) -> LocalizationType | None:
    update: str | None = options.get("UPDATE")
    if update is None:
        return LocalizationType.GLOBAL

    normalized_update = update.upper()
    if normalized_update == "FALSE":
        return None
    if normalized_update == "TRUE":
        return LocalizationType.GLOBAL

    raise ConfigValidationError(f"Invalid UPDATE option: {update}")
