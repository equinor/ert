import os
from typing import Dict, List

from jinja2 import Template

import ert
from ert.shared import ert_share_path


def _get_jobs_from_directories(directories: List[str]) -> Dict[str, str]:
    share_path = ert_share_path()
    directories = [
        Template(directory).render(ERT_SHARE_PATH=share_path, ERT_UI_MODE="gui")
        for directory in directories
    ]

    all_files = []
    for directory in directories:
        all_files.extend(
            [
                os.path.join(directory, f)
                for f in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, f))
            ]
        )
    return {os.path.basename(path): path for path in all_files}


def _get_file_content_if_exists(file_name: str, default: str = "") -> str:
    if os.path.isfile(file_name):
        with open(file_name, encoding="utf-8") as fh:
            return fh.read()
    return default


def _get_job_category(job_name: str) -> str:
    if "FILE" in job_name or "DIR" in job_name or "SYMLINK" in job_name:
        return "utility.file_system"

    if job_name in {"ECLIPSE100", "ECLIPSE300", "FLOW"}:
        return "simulators.reservoir"

    if job_name == "TEMPLATE_RENDER":
        return "utility.templating"

    return "other"


@ert.plugin(name="ert")
def installable_workflow_jobs() -> Dict[str, str]:
    directories = [
        "{{ERT_SHARE_PATH}}/workflows/jobs/shell",
        "{{ERT_SHARE_PATH}}/workflows/jobs/internal-{{ERT_UI_MODE}}/config",
    ]
    return _get_jobs_from_directories(directories)
