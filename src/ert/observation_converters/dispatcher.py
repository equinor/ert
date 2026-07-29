from collections.abc import Callable
from enum import StrEnum
from typing import Any

from ert.cli.main import ErtCliError
from ert.namespace import Namespace

from .history_to_summary import convert_history_to_summary
from .summary_to_bulk import (
    convert_summary_to_bulk,
)


class SupportedFormat(StrEnum):
    SUMMARY = "summary"
    BULK = "bulk"


ConverterFunction = Callable[[str], None]

_SUPPORTED_CONVERSIONS: dict[SupportedFormat, ConverterFunction] = {
    SupportedFormat.BULK: convert_summary_to_bulk,
    SupportedFormat.SUMMARY: convert_history_to_summary,
}


def convert_observations(args: Namespace, _site_plugins: Any | None = None) -> None:
    converter_func = _SUPPORTED_CONVERSIONS.get(args.format)

    if converter_func is None:
        supported_formats = "\n".join(_SUPPORTED_CONVERSIONS.keys())
        raise ErtCliError(
            f"Unsupported format to convert to: {args.format}\n"
            f"Supported formats:\n"
            f"{supported_formats}"
        )

    converter_func(args.config)
