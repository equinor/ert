import os

import pkg_resources
from ert.shared import ert_share_path

from everest.jobs import io, templating, well_tools

from .recovery_factor import recovery_factor

script_names = (
    "render",
    "recovery_factor",
    "wdreorder",
    "wdfilter",
    "wdupdate",
    "wdset",
    "wdcompl",
    "wddatefilter",
)

__all__ = [
    "recovery_factor",
    "get_system_installed_shell_jobs",
    "io",
    "templating",
    "well_tools",
    "script_names",
]


def get_system_installed_shell_jobs():
    try:
        path = os.path.join(ert_share_path(), "forward-models/shell")
        shell_scripts = [
            job_name
            for job_name in os.listdir(path)
            if os.path.isfile(os.path.join(path, job_name))
        ]
        return shell_scripts
    except OSError:
        return (
            "careful_copy_file",
            "copy_directory",
            "copy_file",
            "delete_directory",
            "delete_file",
            "make_directory",
            "make_symlink",
            "move_file",
            "symlink",
        )


def _inject_scripts():
    def fetch_script_path(script_name):
        rel_script_path = os.path.join("scripts", script_name)
        return pkg_resources.resource_filename("everest.jobs", rel_script_path)

    _scripts = {}
    for script_name in script_names:
        _scripts[script_name] = fetch_script_path(script_name)
        globals()[script_name] = _scripts[script_name]

    globals()["_scripts"] = _scripts


def fetch_script(script_name):
    if script_name in _scripts:  # noqa F821
        return _scripts[script_name]  # noqa F821
    else:
        raise KeyError("Unknown script: %s" % script_name)


_inject_scripts()
shell_commands = get_system_installed_shell_jobs()
