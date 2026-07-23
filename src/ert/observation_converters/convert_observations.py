from collections.abc import Callable
from enum import StrEnum
from typing import Any

from ert.namespace import Namespace

from .history_to_summary import run_convert_observations
from .summary_to_bulk_config import (
    convert_summary_to_bulk_config,
)


class SupportedFormats(StrEnum):
    SUMMARY = "summary"
    BULK = "bulk"


Conversion = tuple[SupportedFormats, SupportedFormats]
ConverterFunction = Callable[[str], None]

_SUPPORTED_CONVERSIONS: dict[SupportedFormats, ConverterFunction] = {
    SupportedFormats.BULK: convert_summary_to_bulk_config,
    SupportedFormats.SUMMARY: run_convert_observations,
}
SUPPORTED_FORMATS = _SUPPORTED_CONVERSIONS.keys()


def convert_observations(args: Namespace, _site_plugins: Any | None = None) -> None:
    converter_func = _SUPPORTED_CONVERSIONS.get(args.format)

    if converter_func is None:
        supported_formats = "\n".join(SUPPORTED_FORMATS)
        raise ValueError(
            f"Unsupported format to convert to: {args.format}\n"
            f"Supported formats:\n"
            f"{supported_formats}"
        )

    converter_func(args.config)
