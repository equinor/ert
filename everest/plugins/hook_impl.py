from ert.shared import ert_share_path

from everest.plugins import hookimpl


@hookimpl
def visualize_data(api):
    print("No visualization plugin installed!")
    return None


@hookimpl
def install_job_directories():
    _ert_share_path = ert_share_path()
    job_dirs = [
        "INSTALL_JOB_DIRECTORY      {}/forward-models/shell",
        "INSTALL_JOB_DIRECTORY      {}/forward-models/res",
        "INSTALL_JOB_DIRECTORY      {}/forward-models/templating",
        "INSTALL_JOB_DIRECTORY      {}/forward-models/old_style",
        "",
    ]
    return [line.format(_ert_share_path) for line in job_dirs]


@hookimpl
def default_site_config_lines():
    return [
        "JOB_SCRIPT job_dispatch.py",
        "QUEUE_OPTION LOCAL MAX_RUNNING 1",
        "",
    ]


@hookimpl
def site_config_lines():
    return None


@hookimpl
def ecl100_config_path():
    return None


@hookimpl
def ecl300_config_path():
    return None


@hookimpl
def flow_config_path():
    return None


@hookimpl
def get_forward_models():
    return None


@hookimpl
def get_forward_models_schemas():
    return None


@hookimpl
def installable_workflow_jobs():
    return None
