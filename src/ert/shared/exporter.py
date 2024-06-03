import logging
import re
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from ert.analysis.event import DataSection
from ert.enkf_main import EnKFMain
from ert.gui.ertnotifier import ErtNotifier
from ert.job_queue import WorkflowJobRunner
from ert.libres_facade import LibresFacade

logger = logging.getLogger(__name__)


class Exporter:
    def __init__(self, ert: EnKFMain, notifier: ErtNotifier):
        self.ert = ert
        self.facade = LibresFacade(ert)
        self._export_job = "CSV_EXPORT2"
        self._runpath_job = "EXPORT_RUNPATH"
        self._notifier = notifier

    def is_valid(self) -> bool:
        export_job = self.facade.get_workflow_job(self._export_job)
        runpath_job = self.facade.get_workflow_job(self._runpath_job)

        if export_job is None:
            logger.error(
                f"Export not available because {self._export_job} is not installed."
            )
            return False

        if runpath_job is None:
            logger.error(
                f"Export not available because {self._runpath_job} is not installed."
            )
            return False

        return True

    def run_export(self, parameters: Dict[str, Any]) -> None:
        export_job = self.facade.get_workflow_job(self._export_job)
        if export_job is None:
            raise UserWarning(f"Could not find {self._export_job} job")
        runpath_job = self.facade.get_workflow_job(self._runpath_job)
        if runpath_job is None:
            raise UserWarning(f"Could not find {self._runpath_job} job")

        runpath_job_runner = WorkflowJobRunner(runpath_job)

        runpath_job_runner.run(
            ert=self.ert,
            storage=self._notifier.storage,
            arguments=[],
        )
        if runpath_job_runner.hasFailed():
            raise UserWarning(f"Failed to execute {self._runpath_job}")

        export_job_runner = WorkflowJobRunner(export_job)
        user_warn = export_job_runner.run(
            ert=self.ert,
            storage=self._notifier.storage,
            arguments=[
                str(self.ert.ert_config.runpath_file),
                parameters["output_file"],
                parameters["time_index"],
                parameters["column_keys"],
            ],
        )
        if export_job_runner.hasFailed():
            raise UserWarning(f"Failed to execute {self._export_job}\n{user_warn}")


def csv_event_to_report(name: str, data: DataSection, output_path: Path) -> None:
    fname = str(name).strip().replace(" ", "_")
    fname = re.sub(r"(?u)[^-\w]", "", fname)
    f_path = output_path / fname
    f_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data.data, columns=data.header)
    with open(f_path.with_suffix(".report"), "w", encoding="utf-8") as fout:
        if data.extra:
            for k, v in data.extra.items():
                fout.write(f"{k}: {v}\n")
        fout.write(df.to_markdown(tablefmt="simple_outline", floatfmt=".4f"))
    df.to_csv(f_path.with_suffix(".csv"))
