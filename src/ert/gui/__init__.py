"""The Qt GUI for ert.


The only exported symbol is 'run_gui' which starts the graphical interface.

Note that importing the module loads several heavy graphical libraries,
so this has been done dynamically to avoid incurring overhead when
ert is not running with GUI.

The module is organized around the :py:class:`main window
<ert.gui.main_window.ErtMainWindow>`, which has the :py:class`experiment
panel<ert.gui.experiments.experiment_panel.ExperimentPanel` centrally located
initially. When the user presses "start experiment" this starts an experiment in
the :py:class`run_dialog <ert.gui.experiments.run_dialog.RunDialog>`.

Other situational tools are located in ert.gui.tools, such as plotting results,
and accessed from buttons on the main window.

"""

import os

import matplotlib as mpl


def headless() -> bool:
    return "DISPLAY" not in os.environ


mpl.use("Agg" if headless() else "QtAgg")

from .main import run_gui  # ruff: ignore[module-import-not-at-top-of-file]

__all__ = ["run_gui"]
