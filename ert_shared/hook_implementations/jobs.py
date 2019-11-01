import inspect
import os

from jinja2 import Template
import res

from ert_shared.plugins.plugin_manager import hook_implementation
from ert_shared.plugins.plugin_response import plugin_response


def _resolve_ert_share_path():

    share_path = os.path.realpath(
        os.path.join(os.path.dirname(inspect.getfile(res)), "../../../../share/ert")
    )
    return share_path


def _get_jobs_from_directories(directories):
    share_path = _resolve_ert_share_path()
    directories = list(
        [
            Template(l).render(ERT_SHARE_PATH=share_path, ERT_UI_MODE="gui")
            for l in directories
        ]
    )

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


@hook_implementation
@plugin_response(plugin_name="ert")
def installable_jobs():
    directories = [
        "{{ERT_SHARE_PATH}}/forward-models/shell",
        "{{ERT_SHARE_PATH}}/forward-models/res",
        "{{ERT_SHARE_PATH}}/forward-models/templating",
        "{{ERT_SHARE_PATH}}/forward-models/old_style",
    ]
    return _get_jobs_from_directories(directories)


@hook_implementation
@plugin_response(plugin_name="ert")
def installable_workflow_jobs():
    directories = [
        "{{ERT_SHARE_PATH}}/workflows/jobs/internal/config",
        "{{ERT_SHARE_PATH}}/workflows/jobs/internal-{{ERT_UI_MODE}}/config",
    ]
    return _get_jobs_from_directories(directories)
