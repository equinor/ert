from __future__ import annotations

import logging
import os
from collections import Counter
from importlib.resources import files
from signal import SIG_DFL, SIGINT, signal

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
from ert.storage import ErtStorageException, Storage, open_storage
from ert.storage.local_storage import local_storage_set_ert_config

from .suggestor import Suggestor


def run_gui(args: Namespace, plugin_manager: ErtPluginManager | None = None) -> int:
    # Replace Python's exception handler for SIGINT with the system default.
    #
    # Python's SIGINT handler is the one that raises KeyboardInterrupt. This is
    # okay normally (if a bit ugly), but when control is given to Qt this
    # exception handler will either get deadlocked because Python never gets
    # control back, or gets eaten by Qt because it ignores exceptions that
    # happen in Qt slots.
    signal(SIGINT, SIG_DFL)

    QDir.addSearchPath("img", str(files("ert.gui").joinpath("resources/gui/img")))

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

    storage = None
    if ert_config is not None:
        try:
            storage = open_storage(ert_config.ens_path, mode="w")
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

    assert storage is not None
    main_window = _setup_main_window(
        ert_config, args, log_handler, storage, plugin_manager
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
    storage: Storage,
    plugin_manager: ErtPluginManager | None = None,
) -> ErtMainWindow:
    # window reference must be kept until app.exec returns:
    window = ErtMainWindow(args.config, ert_config, plugin_manager, log_handler)
    window.notifier.set_storage(storage)
    window.post_init()
    window.adjustSize()
    return window
