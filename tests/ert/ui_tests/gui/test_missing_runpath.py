import stat
from contextlib import suppress

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QLabel,
)

from ert.gui.simulation.run_dialog import RunDialog
from ert.run_models import EnsembleExperiment

from .conftest import open_gui_with_config, wait_for_child

config_contents = """\
QUEUE_SYSTEM {queue_system}
NUM_REALIZATIONS 10
LOAD_WORKFLOW_JOB CHMOD_JOB CHMOD
LOAD_WORKFLOW CHMOD.wf CHMOD.wf
HOOK_WORKFLOW CHMOD.wf PRE_SIMULATION

"""

workflow_contents = """\
CHMOD
"""

workflow_job_contents = """\
EXECUTABLE chmod.sh
"""

chmod_sh_contents = """\
#!/bin/bash
chmod 000 {tmp_path}/simulations/realization-0/iter-0
"""


def write_config(tmp_path, queue_system):
    (tmp_path / "config.ert").write_text(
        config_contents.format(queue_system=queue_system)
    )
    (tmp_path / "CHMOD_JOB").write_text(workflow_job_contents)
    (tmp_path / "CHMOD.wf").write_text(workflow_contents)
    (tmp_path / "chmod.sh").write_text(chmod_sh_contents.format(tmp_path=tmp_path))
    (tmp_path / "chmod.sh").chmod(
        (tmp_path / "chmod.sh").stat().st_mode
        | stat.S_IXUSR
        | stat.S_IXGRP
        | stat.S_IXOTH
    )


def test_missing_runpath_has_isolated_failures(
    tmp_path, run_experiment, qtbot, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    write_config(tmp_path, "LOCAL")

    def handle_message_box(dialog):
        def inner():
            qtbot.waitUntil(
                lambda: dialog.fail_msg_box is not None,
                timeout=20000,
            )

            message_box = dialog.fail_msg_box
            assert message_box is not None
            assert message_box.label_text.text() == "ERT experiment failed!"
            message_box.accept()

        return inner

    try:
        for gui in open_gui_with_config(tmp_path / "config.ert"):
            qtbot.addWidget(gui)
            run_experiment(EnsembleExperiment, gui, click_done=False)
            run_dialog = wait_for_child(gui, qtbot, RunDialog, timeout=10000)

            QTimer.singleShot(100, handle_message_box(run_dialog))
            qtbot.waitUntil(
                lambda dialog=run_dialog: not dialog.done_button.isHidden(),
                timeout=200000,
            )
            assert (
                "9/10"
                in run_dialog._progress_widget.findChild(
                    QLabel, name="progress_label_text_Finished"
                ).text()
            )
            assert (
                "1/10"
                in run_dialog._progress_widget.findChild(
                    QLabel, name="progress_label_text_Failed"
                ).text()
            )
            qtbot.mouseClick(run_dialog.done_button, Qt.LeftButton)
    finally:
        with suppress(FileNotFoundError):
            (tmp_path / "simulations/realization-0/iter-0").chmod(0x777)
