from __future__ import annotations

import logging
from collections.abc import Callable
from enum import StrEnum

from ert.cli.main import ErtCliError
from ert.namespace import Namespace
from ert.plugins import ErtRuntimePlugins

from .history_to_summary import convert_history_to_summary
from .summary_to_bulk import (
    convert_summary_to_bulk,
)
from .summary_to_yaml import convert_summary_to_yaml

logger = logging.getLogger(__name__)


class SupportedFormat(StrEnum):
    SUMMARY = "summary"
    BULK = "bulk"
    YAML = "yaml"


ConverterFunction = Callable[[str, ErtRuntimePlugins], None]

_SUPPORTED_CONVERSIONS: dict[SupportedFormat, ConverterFunction] = {
    SupportedFormat.BULK: convert_summary_to_bulk,
    SupportedFormat.SUMMARY: convert_history_to_summary,
    SupportedFormat.YAML: convert_summary_to_yaml,
}


def convert_observations(args: Namespace, site_plugins: ErtRuntimePlugins) -> None:
    converter_func = _SUPPORTED_CONVERSIONS.get(args.format)

    if converter_func is None:
        supported_formats = "\n".join(_SUPPORTED_CONVERSIONS.keys())
        raise ErtCliError(
            f"Unsupported format to convert to: {args.format}\n"
            f"Supported formats:\n"
            f"{supported_formats}"
        )

    logger.info(f"Observation converter used with format: '{args.format}'")
    converter_func(args.config, site_plugins)
