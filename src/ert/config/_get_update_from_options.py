from typing import Literal, get_args

from .parsing import ConfigValidationError

StrategyName = Literal["DISTANCE", "ADAPTIVE", "GLOBAL"]


def get_update_from_options(
    options: dict[str, str], default: str | None = None
) -> str | None:

    update: str | None = options.get("UPDATE", default)
    if update is None:
        return None

    normalized_update = update.upper()
    if normalized_update in {"NONE", "FALSE"}:
        return None
    elif normalized_update == "TRUE":
        return "GLOBAL"

    if normalized_update not in get_args(StrategyName):
        raise ConfigValidationError(
            f"Unknown UPDATE value: {update}. "
            f"Expected one of: {get_args(StrategyName)}, "
            "TRUE, FALSE, or NONE."
        )

    return normalized_update
