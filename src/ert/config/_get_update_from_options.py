from typing import Literal

from ert.config.parsing.config_errors import ConfigValidationError

StrategyName = Literal["DISTANCE", "ADAPTIVE", "GLOBAL"]


def get_update_from_options(options: dict[str, str]) -> str | None:

    update: str | None = options.get("UPDATE")
    if update is None:
        return "GLOBAL"

    normalized_update = update.upper()
    if normalized_update == "FALSE":
        return None
    if normalized_update == "TRUE":
        return "GLOBAL"

    raise ConfigValidationError(f"Invalid UPDATE option: {update}")
