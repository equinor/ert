from __future__ import annotations

import logging
import os
import sys
import time
import traceback
import types
from collections import Counter
from importlib.resources import files
from signal import SIG_DFL, SIGINT, signal

from opentelemetry.trace import Status, StatusCode
from PyQt6.QtCore import QDir
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QWidget

from ert.config import (
    ErrorInfo,
    ErtConfig,
    capture_validation,
)
from ert.gui.main_window import ErtMainWindow
from ert.gui.tools.event_viewer import (
    GUILogHandler,
    add_gui_log_handler,
)
from ert.namespace import Namespace
from ert.plugins import ErtPluginManager
from ert.services import StorageService
from ert.storage import ErtStorageException, open_storage
from ert.storage.local_storage import local_storage_set_ert_config
from ert.trace import trace, tracer

from .suggestor import Suggestor

logger = logging.getLogger(__name__)


@tracer.start_as_current_span("ert.application.gui")
def run_gui(args: Namespace, plugin_manager: ErtPluginManager | None = None) -> int:
    span = trace.get_current_span()
    # Replace Python's exception handler for SIGINT with the system default.
    #
    # Python's SIGINT handler is the one that raises KeyboardInterrupt. This is
    # okay normally (if a bit ugly), but when control is given to Qt this
    # exception handler will either get deadlocked because Python never gets
    # control back, or gets eaten by Qt because it ignores exceptions that
    # happen in Qt slots.
    signal(SIGINT, SIG_DFL)

    QDir.addSearchPath("img", str(files("ert.gui").joinpath("resources/gui/img")))

    sys._excepthook = sys.excepthook  # type: ignore

    def custom_exception_hook(
        exctype: type[BaseException],
        value: BaseException,
        tb: types.TracebackType | None,
    ) -> None:
        """A custom exception hook is needed in order to fully propagate spans and logs
        through OpenTelemetry when exceptions happen in the Qt event loop.

        Note: Any exception occuring in this function will deadlock the application."""
        span.set_status(Status(StatusCode.ERROR))
        try:
            span.record_exception(value)
        except RuntimeError:
            # At least a NameError will yield a non-computable traceback
            value.__traceback__ = None
            span.record_exception(value)
        span.end()

        trace.get_tracer_provider().force_flush()  # type: ignore
        try:
            traceback_str = traceback.format_exception(exctype, value, tb)
        except RuntimeError:
            traceback_str = None

        logger.exception(f"ERT GUI crashed unexpectedly with: {value}\n{traceback_str}")  # noqa: LOG004

        def recursive_logger_flush(logger: logging.Logger) -> None:
            for handler in logger.handlers:
                handler.flush()
            if logger.parent is not None:
                recursive_logger_flush(logger.parent)

        recursive_logger_flush(logger)

        # Pass on exception to original exception handler and exit
        sys._excepthook(exctype, value, traceback)  # type: ignore

        time.sleep(2)  # This gives a cleaner shutdown, but not required for logging
        os._exit(1)

    sys.excepthook = custom_exception_hook

    app = QApplication(["ert"])  # Early so that QT is initialized before other imports
    app.setWindowIcon(QIcon("img:ert_icon.svg"))

    with add_gui_log_handler() as log_handler:
        window, ens_path = _start_initial_gui_window(args, log_handler, plugin_manager)

        def show_window() -> int:
            window.show()
            window.activateWindow()
            window.raise_()
            return app.exec()

        # ens_path is None indicates that there was an error in the setup and
        # window is now just showing that error message, in which
        # case display it and don't show an error message
        if ens_path is None:
            return show_window()

        with StorageService.init_service(project=os.path.abspath(ens_path)):
            return show_window()
    return -1


def _start_initial_gui_window(
    args: Namespace,
    log_handler: GUILogHandler,
    plugin_manager: ErtPluginManager | None = None,
) -> tuple[QWidget, str | None]:
    # Create logger inside function to make sure all handlers have been added to
    # the root-logger.
    logger = logging.getLogger(__name__)
    ert_config = None

    with capture_validation() as validation_messages:
        ert_dir = os.path.abspath(os.path.dirname(args.config))
        os.chdir(ert_dir)
        # Changing current working directory means we need to update
        # the config file to be the base name of the original config
        args.config = os.path.basename(args.config)

        ert_config = ErtConfig.with_plugins().from_file(args.config)

        local_storage_set_ert_config(ert_config)

    storage_path = None
    if ert_config is not None:
        try:
            # Open write to initialize the storage,so that
            # dark storage can be mounted onto it
            open_storage(ert_config.ens_path, mode="w").close()
            storage_path = ert_config.ens_path
        except ErtStorageException as err:
            validation_messages.errors.append(
                ErrorInfo(f"Error opening storage in ENSPATH: {err}").set_context(
                    ert_config.ens_path
                )
            )
    if validation_messages.errors:
        logger.info(f"Error in config file shown in gui: {validation_messages.errors}")
        return (
            Suggestor(
                validation_messages.errors,
                validation_messages.warnings,
                validation_messages.deprecations,
                None,
                (plugin_manager.get_help_links() if plugin_manager is not None else {}),
                widget_info="""\
                    <p style="font-size: 28px;">Some errors detected</p>
                    <p style="font-size: 16px;">The following errors were detected
                    while reading the ert configuration file. </p>
                """,
            ),
            None,
        )
    counter_fm_steps = Counter(fms.name for fms in ert_config.forward_model_steps)

    for fm_step_name, count in counter_fm_steps.items():
        logger.info(
            f"Config contains forward model step {fm_step_name} {count} time(s)",
        )

    for msg in validation_messages.deprecations:
        logger.info(f"Suggestion shown in gui '{msg}'")
    for msg in validation_messages.warnings:
        logger.info(f"Warning shown in gui '{msg}'")

    assert storage_path is not None
    main_window = _setup_main_window(
        ert_config, args, log_handler, storage_path, plugin_manager
    )

    if validation_messages.warnings or validation_messages.deprecations:

        def continue_action() -> None:
            main_window.show()
            main_window.activateWindow()
            main_window.raise_()
            main_window.adjustSize()

        suggestor = Suggestor(
            validation_messages.errors,
            validation_messages.warnings,
            validation_messages.deprecations,
            continue_action,
            plugin_manager.get_help_links() if plugin_manager is not None else {},
            widget_info="""\
                <p style="font-size: 28px;">Some problems detected</p>
                <p style="font-size: 16px;">The following problems were detected
                while reading the ert configuration file. </p>
            """,
        )
        return (
            suggestor,
            ert_config.ens_path,
        )
    else:
        return (
            main_window,
            ert_config.ens_path,
        )


def _setup_main_window(
    ert_config: ErtConfig,
    args: Namespace,
    log_handler: GUILogHandler,
    storage_path: str,
    plugin_manager: ErtPluginManager | None = None,
) -> ErtMainWindow:
    # window reference must be kept until app.exec returns:
    window = ErtMainWindow(args.config, ert_config, plugin_manager, log_handler)
    window.notifier.set_storage(storage_path)
    window.post_init()
    window.adjustSize()
    return window
