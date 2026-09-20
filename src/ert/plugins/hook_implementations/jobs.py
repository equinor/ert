from pathlib import Path

from jinja2 import Template

import ert
from ert.shared import ert_share_path


def _get_jobs_from_directories(directories: list[str]) -> dict[str, str]:
    share_path = ert_share_path()
    rendered_directories = [
        Path(Template(directory).render(ERT_SHARE_PATH=share_path, ERT_UI_MODE="gui"))
        for directory in directories
    ]

    all_files = []
    for directory in rendered_directories:
        all_files.extend([file for file in directory.iterdir() if file.is_file()])
    return {path.name: str(path) for path in all_files}


@ert.plugin(name="ert")
def installable_workflow_jobs() -> dict[str, str]:
    directories = [
        "{{ERT_SHARE_PATH}}/workflows/jobs/shell",
    ]
    return _get_jobs_from_directories(directories)
