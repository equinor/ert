import logging

from opentelemetry.sdk.trace.export import BatchSpanProcessor

from ert.plugins.plugin_manager import hook_specification


@hook_specification
def add_log_handle_to_root() -> logging.Handler:  # type: ignore
    """
    Create a log handle which will be added to the root logger
    in the main entry point.

    :return: A log handle that will be added to the root logger
    """


@hook_specification
def add_span_processor() -> BatchSpanProcessor:  # type: ignore
    """
    Create a BatchSpanProcessor which will be added to the trace provider
    in ert.

    :return: A BatchSpanProcessor that will be added to the trace provider in ert
    """
