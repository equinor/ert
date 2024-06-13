import errno
import os
import tempfile
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from pydantic import ValidationError

from everest.config.install_data_config import InstallDataConfig
from everest.util.forward_models import (
    collect_forward_model_schemas,
    parse_forward_model_file,
)

from .install_job_config import InstallJobConfig

if TYPE_CHECKING:
    from pydantic_core import ErrorDetails


class InstallDataContext:
    def __init__(self, install_data: List[InstallDataConfig], config_path: Path):
        self._install_data = install_data or []
        self._config_dir = str(config_path.parent)
        self._cwd = os.getcwd()

    def __enter__(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        for data in self._install_data:
            if "<GEO_ID>" not in data.source:
                self._set_symlink(data.source, data.target, None)

        os.chdir(self._temp_dir.name)
        return self

    def _set_symlink(self, source: str, target: str, realization: Optional[int]):
        if realization is not None:
            source = source.replace("<GEO_ID>", str(realization))
            target = target.replace("<GEO_ID>", str(realization))

        tmp_target = Path(self._temp_dir.name) / Path(target)
        if tmp_target.exists():
            tmp_target.unlink()
        tmp_target.parent.mkdir(parents=True, exist_ok=True)
        tmp_target.symlink_to(as_abs_path(source, self._config_dir))

    def add_links_for_realization(self, realization: int):
        for data in self._install_data:
            if "<GEO_ID>" in data.source:
                self._set_symlink(data.source, data.target, realization)

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self._temp_dir:
            self._temp_dir.cleanup()
        os.chdir(self._cwd)


def check_path_valid(path: str):
    if not isinstance(path, str):
        raise ValueError("str type expected")

    root = os.path.sep
    for dirname in str(path).split(os.path.sep):
        try:
            os.lstat(os.path.join(root, dirname))

        except OSError as e:
            if e.errno in (errno.ENAMETOOLONG, errno.ERANGE):
                raise ValueError(e.strerror) from e
        except TypeError as e:
            raise ValueError(str(e)) from e


def check_writable_filepath(path: str):
    check_path_valid(path)

    if os.path.isdir(path) or not os.path.basename(path):
        raise ValueError("Invalid type")
    if os.path.isfile(path) and not os.access(path, os.W_OK | os.X_OK):
        raise ValueError(f"User does not have write access to {path}")


def check_for_duplicate_names(names: List[str], item_name: str, key: str = "item"):
    if len(set(names)) != len(names):
        histogram = {k: names.count(k) for k in set(names) if names.count(k) > 1}
        occurrences_str = ", ".join(
            [f"{k} ({v} occurrences)" for k, v in histogram.items()]
        )
        raise ValueError(
            f"{item_name.capitalize()} {key}s must be unique. Detected multiple "
            f"occurrences  of the following"
            f" {item_name.lower()} {key}: {occurrences_str}"
        )


def as_abs_path(path: str, config_dir: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.realpath(os.path.join(config_dir, path))


def expand_geo_id_paths(path_source: str, realizations: List[int]):
    if "<GEO_ID>" in path_source:
        return [path_source.replace("<GEO_ID>", str(r)) for r in realizations]
    return [path_source]


def check_path_exists(
    path_source: str, config_path: Optional[Path], realizations: List[int]
):
    """Check if the given path exists. If the given path contains <CONFIG_PATH>
    or GEO_ID they will be expanded and all instances of expanded paths need to exist.
    """
    if not isinstance(path_source, str):
        raise ValueError(
            f"Expected path_source to be a str, but got {type(path_source)}"
        )
    if config_path is None:
        raise ValueError("Config path not defined")
    config_dir = config_path.parent

    if path_source.count("<CONFIG_PATH>") > 1:
        raise ValueError(f"<CONFIG_PATH> occurs twice in path {path_source}.")
    if path_source.count("<CONFIG_PATH>") == 1:
        pre, pos = path_source.split("<CONFIG_PATH>")
        if pos and pos[0] == "/":
            pos = pos[1:]
            path_source = os.path.join(pre, config_dir, pos)

    expanded_paths = expand_geo_id_paths(str(path_source), realizations)
    for exp_path in [as_abs_path(p, str(config_dir)) for p in expanded_paths]:
        if not os.path.exists(exp_path):
            raise ValueError(f"No such file or directory {exp_path}")


def check_writeable_path(path_source: str, config_path: Path):
    # check that the lowest existing folder is writeable
    path = as_abs_path(path_source, str(config_path.parent))
    while True:
        if os.path.isdir(path):
            if os.access(path, os.W_OK | os.X_OK):
                break
        elif os.path.isfile(path):
            # path is a file, cannot create folder
            raise ValueError(f"File {path} exists, cannot create path {path_source}")
        parent = os.path.dirname(path)
        if parent == path:  # ie, if path is root
            break
        path = parent

    if not os.access(path, os.W_OK | os.X_OK):
        raise ValueError(f"User does not have write access to {path}")


def _error_loc(error_dict: "ErrorDetails") -> str:
    return " -> ".join(
        str(e) for e in error_dict["loc"] if e is not None and e != "__root__"
    )


def format_errors(error: ValidationError) -> str:
    errors = error.errors()
    msg = f"Found  {len(errors)} validation error{'s' if len(errors) > 1 else ''}:\n\n"
    error_map = {}
    for err in error.errors():
        key = _error_loc(err)
        if key not in error_map:
            error_map[key] = [key]
        error_map[key].append(f"    * {err['msg']} (type={err['type']})")
    return msg + "\n".join(list(chain.from_iterable(error_map.values())))


def validate_forward_model_configs(
    forward_model: Optional[List[str]], install_jobs: Optional[List[InstallJobConfig]]
):
    if not forward_model:
        return

    install_jobs = install_jobs or []
    user_defined_jobs = [job.name for job in install_jobs]

    def _job_config_path():
        for i, arg in enumerate(args):
            if arg in ["-c", "--config"]:
                if i + 1 >= len(args):
                    return None
                return args[i + 1]

    job_schemas = collect_forward_model_schemas()

    for command in forward_model:
        job, *args = command.split()
        if job in user_defined_jobs:
            continue
        path = _job_config_path()
        schema = job_schemas.get(job)
        if schema is None:
            continue
        if path is None:
            raise ValueError(f"No config file specified for job {job}")
        message = f"{job = }\t-c/--config = {path}\n\t\t{{error}}"
        parse_forward_model_file(path, schema, message)
