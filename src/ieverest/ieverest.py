from pydantic import ValidationError
from qtpy.QtCore import QObject, QThread, Signal
from qtpy.QtWidgets import QFileDialog, QMessageBox, qApp

from ert.config import ErtConfig
from ert.storage import open_storage
from everest.config import EverestConfig, validation_utils
from everest.detached import (
    ServerStatus,
    everserver_status,
    generate_everserver_ert_config,
    get_opt_status,
    get_sim_status,
    server_is_running,
    start_monitor,
    start_server,
    stop_server,
    wait_for_context,
    wait_for_server,
    wait_for_server_to_stop,
)
from everest.plugins.site_config_env import PluginSiteConfigEnv
from everest.simulator import Status
from everest.strings import OPT_PROGRESS_ID, SIM_PROGRESS_ID
from everest.util import configure_logger, makedirs_if_needed
from ieverest import settings
from ieverest.io import QtDialogsOut, QtStatusBarOut
from ieverest.main_window import MainWindow
from ieverest.utils import (
    APP_OUT_DIALOGS,
    APP_OUT_LOGGER,
    APP_OUT_STATUS_BAR,
    app_output,
)
from ieverest.widgets import ExportDataDialog


class ServerMonitorThread(QThread):
    def __init__(self, config: EverestConfig):
        self._config = config
        super(ServerMonitorThread, self).__init__()
        self.stop = False

    progress = Signal(dict)
    finished = Signal()

    @staticmethod
    def convert(status):
        transformed_signal = {}
        if SIM_PROGRESS_ID in status:
            sim_progress = status[SIM_PROGRESS_ID]
            if sim_progress:
                sim_progress.update({"status": Status(**sim_progress["status"])})
                transformed_signal.update(
                    {
                        "type": SIM_PROGRESS_ID,
                        "status": sim_progress,
                    }
                )

        if OPT_PROGRESS_ID in status:
            opt_progress = status[OPT_PROGRESS_ID]
            if opt_progress:
                transformed_signal.update(
                    {
                        "type": OPT_PROGRESS_ID,
                        "progress": opt_progress,
                    }
                )

        return transformed_signal

    def signal_callback(self, status):
        converted_status = self.convert(status)
        self.progress.emit(converted_status)
        return self.stop

    def run(self):
        start_monitor(config=self._config, callback=self.signal_callback)
        # Emit finished signal when monitoring is done
        self.finished.emit()

    def quit(self):
        self.stop = True
        super(ServerMonitorThread, self).quit()


class IEverest(QObject):
    """Main class of the graphical version of Everest.

    This class has the following responsibilities:
    - managing an Everest configuration
    - setting up the GUI
    - controlling the Everest optimization backend
    """

    def __init__(self, config_file=None, parent=None):
        super(IEverest, self).__init__(parent)
        self._gui = MainWindow()
        self._setup_connections()
        self._storage = None
        aout = app_output()
        gui_logger = configure_logger(APP_OUT_LOGGER, log_to_azure=True)
        aout.add_output_channel(gui_logger, APP_OUT_LOGGER)
        aout.add_output_channel(QtDialogsOut(self._gui), APP_OUT_DIALOGS)
        aout.add_output_channel(
            QtStatusBarOut(self._gui.statusBar()), APP_OUT_STATUS_BAR
        )

        self._gui.recent_files = settings.recent_files()
        self._gui.show()

        self.server_monitor = None
        # This needs to be the last part of the IEverest constructor
        if config_file is not None:
            self.open(config_file)

    @property
    def config(self) -> EverestConfig:
        return self._gui.get_config()

    @config.setter
    def config(self, new_config: EverestConfig):
        self._gui.set_config(new_config)

    def _setup_connections(self):
        """Connect the signals coming from the GUI to appropriate slots"""
        self._gui.export_requested.connect(self.export)
        self._gui.open_requested.connect(self.open)
        self._gui.quit_requested.connect(self.quit)

        self._gui.monitor_gui.start_opt_requested.connect(self.start_optimization)
        self._gui.monitor_gui.stop_opt_requested.connect(self.stop_optimization_prompt)
        self._gui.monitor_gui.export_requested.connect(self.export)

    def stop_optimization_prompt(self):
        """Executes and returns the user preference with respect to killing the
        server"""
        status = self.stop_monitoring()
        self._gui.monitor_gui.update_status(
            workflow_is_running=not status,
        )
        return status

    def _update_recent_files(self, filepath):
        files = settings.add_recent_file(filepath)
        self._gui.recent_files = files

    def export(self):
        if server_is_running(self.config):
            app_output().info("Cannot export while optimization is running")
            return
        export_data_dlg = ExportDataDialog(self.config)
        export_data_dlg.show()

    def open(self, filename=""):
        if server_is_running(self.config) and not self.stop_optimization_prompt():
            return
        filename = str(filename)  # Convert Qt4 QString to str
        if not filename:
            selection = QFileDialog.getOpenFileName(
                self._gui,
                "Open configuration",
                (
                    str(self.config.config_path)
                    if self.config.config_path is not None
                    else ""
                ),
                "Everest configuration (*.yml);;All files (*.*)",
            )
            filename = str(selection[0])
            if not filename:
                return  # user cancelled
        try:
            opened_config = EverestConfig.load_file(filename)
        except FileNotFoundError:
            app_output().critical(f"File not found {filename}")
            return
        except ValidationError as e:
            app_output().critical(
                f"Loading config file <{filename}> failed with:\n\n"
                f"{validation_utils.format_errors(e)}"
            )
            app_output().info(validation_utils.format_errors(e))
            return
        self.config = opened_config
        self._update_recent_files(filename)
        server_state = everserver_status(self.config)

        if server_is_running(self.config):
            self._gui.monitor_gui.update_status(
                workflow_is_running=True,
            )
            ert_config = ErtConfig.with_plugins().from_dict(
                generate_everserver_ert_config(self.config)
            )
            self._storage = open_storage(ert_config.ens_path)
            self.start_monitoring_detached_server(self.update_progress_callback)

            # Detached server found maybe change some buttons
            app_output().info("Attaching to optimization session in progress...")
        elif server_state["status"] != ServerStatus.never_run:
            self._gui.monitor_gui.reset()
            self._gui.monitor_gui.update_optimization_progress(
                get_opt_status(self.config.optimization_output_dir)
            )
            self._gui.monitor_gui.update_status(get_sim_status(self.config))

        else:
            self._gui.monitor_gui.reset()

    def stop_monitoring(self, timeout=60):
        msg = """
        A detached optimization is currently running.\n
        Do you want to stop it?
        {}""".format(
            """Press Yes to stop the optimization
        Press No to only stop monitoring the optimization
        """
            if self.server_monitor is not None
            else ""
        )

        answer = QMessageBox.question(
            self._gui,
            "Stop detached optimization?",
            msg,
            QMessageBox.Cancel | QMessageBox.No | QMessageBox.Yes,
        )
        if answer == QMessageBox.Yes:
            if stop_server(self.config):
                wait_for_server_to_stop(self.config, timeout)
                self.stop_monitoring_detached_server()
                return True
            return False
        if answer == QMessageBox.No:
            self.stop_monitoring_detached_server()
            return True
        return False

    def quit(self):
        if not server_is_running(self.config) or self.stop_optimization_prompt():
            qApp.quit()
        else:
            return

    def update_progress_callback(self, prog):
        if not prog or "type" not in prog:
            return None

        if prog["type"] == SIM_PROGRESS_ID:
            self._gui.monitor_gui.update_status(
                workflow_status=prog["status"],
                workflow_is_running=self.server_monitor is not None,
            )
        else:
            assert prog["type"] == OPT_PROGRESS_ID
            self._gui.monitor_gui.update_optimization_progress(prog["progress"])
        return True

    def start_optimization(self):
        """Run an optimization using the current configuration"""
        self._gui.monitor_gui.update_status(workflow_is_running=True)
        app_output().info("Starting optimization...")

        config_dict = self.config.to_dict()
        app_output().info(f"Running everest with config info\n{config_dict!s}")
        for fm_job in self.config.forward_model:
            job_name = fm_job.split()[0]
            app_output().info(f"Everest forward model contains job {job_name}")

        server_started = self.start_detached_server()
        if server_started:
            self.start_monitoring_detached_server(self.update_progress_callback)

    def start_detached_server(self):
        if not server_is_running(self.config):
            app_output().info("Starting optimization session....")
            with PluginSiteConfigEnv():
                ert_config = ErtConfig.with_plugins().from_dict(
                    generate_everserver_ert_config(self.config)
                )

                makedirs_if_needed(self.config.output_dir, roll_if_exists=True)
                self._storage = open_storage(ert_config.ens_path, "w")
                context = start_server(self.config, ert_config, self._storage)
                try:
                    wait_for_server(self.config, timeout=600, context=context)
                except:
                    app_output().info("Starting session failed!")
                    return False
        return True

    def server_monitoring_finished(self):
        wait_for_context()
        self._gui.monitor_gui.update_status(workflow_is_running=False)
        self.stop_monitoring_detached_server()
        server_state = everserver_status(self.config)

        if server_state["status"] == ServerStatus.completed:
            self._gui.monitor_gui.reset()
            self._gui.monitor_gui.update_optimization_progress(
                get_opt_status(self.config.optimization_output_dir)
            )
            self._gui.monitor_gui.update_status(get_sim_status(self.config))
            app_output().info("Optimization completed")
        elif server_state["status"] == ServerStatus.stopped:
            app_output().info("Optimization stopped")
        elif server_state["status"] == ServerStatus.failed:
            app_output().critical(
                "Optimization failed:\n {}".format(server_state["message"])
            )

    def start_monitoring_detached_server(self, progress_callback):
        if self.server_monitor:
            self.server_monitor.quit()

        self.server_monitor = ServerMonitorThread(config=self.config)

        # Connect monitoring server thread signals
        self.server_monitor.progress.connect(progress_callback)
        self.server_monitor.finished.connect(self.server_monitoring_finished)

        self._gui.monitor_gui.reset()

        self.server_monitor.start()

    def stop_monitoring_detached_server(self):
        if self.server_monitor is not None:
            self.server_monitor.quit()
            self.server_monitor = None
        if self._storage:
            self._storage.close()
