import os

import pkg_resources
from jinja2 import Template

from ert.shared.plugins.plugin_manager import hook_implementation
from ert.shared.plugins.plugin_response import plugin_response


def _resolve_ert_share_path():
    return pkg_resources.resource_filename("ert.shared", "share/ert")


def _get_jobs_from_directories(directories):
    share_path = _resolve_ert_share_path()
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


def _get_file_content_if_exists(file_name, default=""):
    if os.path.isfile(file_name):
        with open(file_name, encoding="utf-8") as fh:
            return fh.read()
    return default


def _get_job_category(job_name):
    if "FILE" in job_name or "DIR" in job_name or "SYMLINK" in job_name:
        return "utility.file_system"

    if job_name in {"ECLIPSE100", "ECLIPSE300", "FLOW"}:
        return "simulators.reservoir"

    if job_name == "RMS":
        return "modelling.reservoir"

    if job_name == "TEMPLATE_RENDER":
        return "utility.templating"

    return "other"


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
def job_documentation(job_name):
    ert_jobs = set(installable_jobs().data.keys())
    if job_name not in ert_jobs:
        return None

    doc_folder = f"{_resolve_ert_share_path()}/forward-models/docs"

    description_file = os.path.join(doc_folder, "description", f"{job_name}.rst")
    description = _get_file_content_if_exists(description_file)

    examples_file = os.path.join(doc_folder, "examples", f"{job_name}.rst")
    examples = _get_file_content_if_exists(examples_file)

    return {
        "description": description,
        "examples": examples,
        "category": _get_job_category(job_name),
    }


@hook_implementation
@plugin_response(plugin_name="ert")
def installable_workflow_jobs():
    directories = [
        "{{ERT_SHARE_PATH}}/workflows/jobs/shell",
        "{{ERT_SHARE_PATH}}/workflows/jobs/internal-{{ERT_UI_MODE}}/config",
    ]
    return _get_jobs_from_directories(directories)
