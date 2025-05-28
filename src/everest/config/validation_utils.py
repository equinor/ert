import errno
import os
import tempfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self, TypeVar

from pydantic import BaseModel

from everest.config.install_data_config import InstallDataConfig
from everest.util.forward_models import (
    collect_forward_model_schemas,
    lint_forward_model_job,
    parse_forward_model_file,
)

from .install_job_config import InstallJobConfig

_VARIABLE_ERROR_MESSAGE = (
    "Variable {name} must define {variable_type} value either"
    " at control level or variable level"
)


class InstallDataContext:
    def __init__(
        self, install_data: list[InstallDataConfig], config_path: Path
    ) -> None:
        self._install_data = install_data or []
        self._config_dir = str(config_path.parent)
        self._cwd = os.getcwd()

    def __enter__(self) -> Self:
        self._temp_dir = tempfile.TemporaryDirectory()
        for data in self._install_data:
            if "<GEO_ID>" not in data.source:
                self._set_symlink(data.source, data.target, None)

        os.chdir(self._temp_dir.name)
        return self

    def _set_symlink(self, source: str, target: str, realization: int | None) -> None:
        if realization is not None:
            source = source.replace("<GEO_ID>", str(realization))
            target = target.replace("<GEO_ID>", str(realization))
        if target.startswith("../"):
            raise ValueError(
                f"Target location outside of runpath {target} not allowed!"
            )

        tmp_target = Path(self._temp_dir.name) / Path(target)
        if tmp_target.exists():
            tmp_target.unlink()
        tmp_target.parent.mkdir(parents=True, exist_ok=True)
        tmp_target.symlink_to(as_abs_path(source, self._config_dir))

    def add_links_for_realization(self, realization: int) -> None:
        for data in self._install_data:
            if "<GEO_ID>" in data.source:
                self._set_symlink(data.source, data.target, realization)

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        if self._temp_dir:
            self._temp_dir.cleanup()
        os.chdir(self._cwd)


def control_variables_validation(
    name: str,
    min_: float | None,
    max_: float | None,
    initial_guess: float | list[float] | None,
) -> list[str]:
    error = []
    if min_ is None:
        error.append(_VARIABLE_ERROR_MESSAGE.format(name=name, variable_type="min"))
    if max_ is None:
        error.append(_VARIABLE_ERROR_MESSAGE.format(name=name, variable_type="max"))
    if initial_guess is None:
        error.append(
            _VARIABLE_ERROR_MESSAGE.format(name=name, variable_type="initial_guess")
        )
    if isinstance(initial_guess, float):
        initial_guess = [initial_guess]
    if (
        min_ is not None
        and max_ is not None
        and (
            msg := ", ".join(
                str(guess) for guess in initial_guess or [] if not min_ <= guess <= max_
            )
        )
    ):
        error.append(
            f"Variable {name} must respect {min_} <= initial_guess <= {max_}: {msg}"
        )
    return error


def no_dots_in_string(value: str) -> str:
    if "." in value:
        raise ValueError("Variable name can not contain any dots (.)")
    return value


T = TypeVar("T", bound=BaseModel)


def _duplicate_string(items: Sequence[T]) -> str:
    def duplicate_values(item: T) -> str:
        return ", ".join(
            f"{key}: {getattr(item, key) or 'null'}"
            for key in item.uniqueness.split("-")  # type: ignore
        )

    return ", ".join(
        f"'{duplicate_values(item)}' ({count} occurrences)"
        for item, count in Counter(items).items()
        if count > 1
    )


def uniform_variables(items: Sequence[T]) -> Sequence[T]:
    if (
        len(
            {
                len(variable.initial_guess)  # type: ignore
                for variable in items
                if isinstance(variable.initial_guess, list)  # type: ignore
            }
        )
        > 1
    ):
        raise ValueError("All initial_guess list must be the same length")
    return items


def unique_items(items: Sequence[T]) -> Sequence[T]:
    if duplicates := _duplicate_string(items):
        raise ValueError(
            f"Subfield(s) `{items[0].uniqueness}` must be unique. "  # type: ignore
            f"Detected multiple occurrences of the following {duplicates}"
        )
    return items


def valid_range(range_value: tuple[float, float]) -> tuple[float, float]:
    if range_value[0] >= range_value[1]:
        raise ValueError("scaled_range must be a valid range [a, b], where a < b.")
    return range_value


def check_path_valid(path: str) -> None:
    if not isinstance(path, str):
        raise ValueError("str type expected")

    root = os.path.sep
    for dirname in str(path).split(os.path.sep):
        try:
            os.lstat(os.path.join(root, dirname))

        except OSError as e:
            if e.errno in {errno.ENAMETOOLONG, errno.ERANGE}:
                raise ValueError(e.strerror) from e
        except TypeError as e:
            raise ValueError(str(e)) from e


def check_writable_filepath(path: str) -> None:
    check_path_valid(path)

    if os.path.isdir(path) or not os.path.basename(path):
        raise ValueError("Invalid type")
    if os.path.isfile(path) and not os.access(path, os.W_OK):
        raise ValueError(f"User does not have write access to {path}")


def check_for_duplicate_names(
    names: list[str], item_name: str, key: str = "item"
) -> None:
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


def expand_geo_id_paths(path_source: str, realizations: list[int]) -> list[str]:
    if "<GEO_ID>" in path_source:
        return [path_source.replace("<GEO_ID>", str(r)) for r in realizations]
    return [path_source]


def check_path_exists(
    path_source: str, config_path: Path | None, realizations: list[int]
) -> None:
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
        if os.path.ismount(exp_path):
            raise ValueError(f"'{exp_path}' is a mount point and can't be handled")
        if not os.path.exists(exp_path):
            raise ValueError(f"No such file or directory {exp_path}")


def check_writeable_path(path_source: str, config_path: Path) -> None:
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


def validate_forward_model_configs(
    forward_model: list[str], install_jobs: list[InstallJobConfig]
) -> None:
    if not forward_model:
        return

    user_defined_jobs = [job.name for job in install_jobs]

    def _job_config_index(*args):  # type: ignore
        return next(
            (
                i
                for i, arg in enumerate(args)
                if arg in {"-c", "--config"} and i + 1 < len(args)
            ),
            None,
        )

    job_schemas = collect_forward_model_schemas()

    for command in forward_model:
        job, *args = command.split()
        if (
            job in user_defined_jobs
            or not args
            or args[0] == "schema"
            or (schema := job_schemas.get(job)) is None  # type: ignore
        ):
            continue

        if (index := _job_config_index(*args)) is None:  # type: ignore
            raise ValueError(f"No config file specified for job {job}")

        if args[0] in {"run", "lint"} and (
            errors := lint_forward_model_job(job, args[index : index + 2])
        ):
            raise ValueError("\n\t".join(errors))

        path = args[index + 1]
        message = f"{job = }\t-c/--config = {path}\n\t\t{{error}}"
        parse_forward_model_file(path, schema, message)
