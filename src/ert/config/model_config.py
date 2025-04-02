from __future__ import annotations

import contextlib
import logging
import os.path
import shutil
from datetime import datetime
from typing import no_type_check

from pydantic import field_validator
from pydantic.dataclasses import dataclass

from ert.shared.status.utils import byte_with_unit, get_mount_directory

from .parsing import (
    ConfigDict,
    ConfigKeys,
    ConfigValidationError,
    ConfigWarning,
    HistorySource,
)

logger = logging.getLogger(__name__)


def _read_time_map(file_contents: str) -> list[datetime]:
    def str_to_datetime(date_str: str) -> datetime:
        try:
            return datetime.fromisoformat(date_str)
        except ValueError:
            logger.warning(
                "DD/MM/YYYY date format is deprecated"
                ", please use ISO date format YYYY-MM-DD."
            )
            return datetime.strptime(date_str, "%d/%m/%Y")

    dates = []
    for line in file_contents.splitlines():
        dates.append(str_to_datetime(line.strip()))
    return dates


DEFAULT_HISTORY_SOURCE = HistorySource.REFCASE_HISTORY
DEFAULT_RUNPATH = "simulations/realization-<IENS>/iter-<ITER>"
DEFAULT_GEN_KW_EXPORT_NAME = "parameters"
DEFAULT_JOBNAME_FORMAT = "<CONFIG_FILE>-<IENS>"
DEFAULT_ECLBASE_FORMAT = "ECLBASE<IENS>"

FULL_DISK_PERCENTAGE_THRESHOLD = 0.97
MINIMUM_BYTES_LEFT_ON_DISK_THRESHOLD = 200 * 1000**3  # 200 GB
# We give warning if free disk space is less than MINIMUM_BYTES_LEFT_ON_DISK_THRESHOLD
# and used space in percentage is greater than FULL_DISK_PERCENTAGE_THRESHOLD


@dataclass
class ModelConfig:
    num_realizations: int = 1
    history_source: HistorySource = DEFAULT_HISTORY_SOURCE
    runpath_format_string: str = DEFAULT_RUNPATH
    jobname_format_string: str = DEFAULT_JOBNAME_FORMAT
    eclbase_format_string: str = DEFAULT_ECLBASE_FORMAT
    gen_kw_export_name: str = DEFAULT_GEN_KW_EXPORT_NAME
    time_map: list[datetime] | None = None

    @field_validator("runpath_format_string", mode="before")
    @classmethod
    def validate_runpath(cls, runpath_format_string: str) -> str:
        if "%d" in runpath_format_string and any(
            x in runpath_format_string for x in ["<ITER>", "<IENS>"]
        ):
            raise ConfigValidationError(
                f"RUNPATH cannot combine deprecated and new style placeholders: `{runpath_format_string}`. Valid example `{DEFAULT_RUNPATH}`"
            )
        # do not allow multiple occurrences
        for kw in ["<ITER>", "<IENS>"]:
            if runpath_format_string.count(kw) > 1:
                raise ConfigValidationError(
                    f"RUNPATH cannot contain multiple {kw} placeholders: `{runpath_format_string}`. Valid example `{DEFAULT_RUNPATH}`"
                )
        # do not allow too many placeholders
        if runpath_format_string.count("%d") > 2:
            raise ConfigValidationError(
                f"RUNPATH cannot contain more than two value placeholders: `{runpath_format_string}`. Valid example `{DEFAULT_RUNPATH}`"
            )
        result = _replace_runpath_format(runpath_format_string)
        if not any(x in result for x in ["<ITER>", "<IENS>"]):
            msg = (
                "RUNPATH keyword contains no value placeholders: "
                f"`{runpath_format_string}`. Valid example: "
                f"`{DEFAULT_RUNPATH}` "
            )
            ConfigWarning.warn(msg)
            logger.warning(msg)
        with contextlib.suppress(Exception):
            mount_dir = get_mount_directory(runpath_format_string)
            total_space, used_space, free_space = shutil.disk_usage(mount_dir)
            percentage_used = used_space / total_space
            if (
                percentage_used > FULL_DISK_PERCENTAGE_THRESHOLD
                and free_space < MINIMUM_BYTES_LEFT_ON_DISK_THRESHOLD
            ):
                msg = (
                    f"Low disk space: {byte_with_unit(free_space)} free on {mount_dir!s}."
                    " Consider freeing up some space to ensure successful simulation runs."
                )
                ConfigWarning.warn(msg)
        return result

    @field_validator("jobname_format_string", mode="before")
    @classmethod
    def validate_jobname(cls, jobname_format_string: str) -> str:
        result = _replace_runpath_format(jobname_format_string)
        if "/" in jobname_format_string:
            raise ConfigValidationError.with_context(
                "JOBNAME cannot contain '/'.", jobname_format_string
            )
        return result

    @field_validator("eclbase_format_string", mode="before")
    @classmethod
    def transform(cls, eclbase_format_string: str) -> str:
        return _replace_runpath_format(eclbase_format_string)

    @no_type_check
    @classmethod
    def from_dict(cls, config_dict: ConfigDict) -> ModelConfig:
        time_map_args = config_dict.get(ConfigKeys.TIME_MAP)
        time_map = None
        if time_map_args is not None:
            time_map_file, time_map_contents = time_map_args
            try:
                time_map = _read_time_map(time_map_contents)
            except ValueError as err:
                raise ConfigValidationError.with_context(
                    f"Could not read timemap file {time_map_file}: {err}", time_map_file
                ) from err
        return cls(
            num_realizations=config_dict.get(ConfigKeys.NUM_REALIZATIONS, 1),
            history_source=config_dict.get(
                ConfigKeys.HISTORY_SOURCE, DEFAULT_HISTORY_SOURCE
            ),
            runpath_format_string=config_dict.get(ConfigKeys.RUNPATH, DEFAULT_RUNPATH),
            jobname_format_string=config_dict.get(
                ConfigKeys.JOBNAME,
                os.path.basename(
                    config_dict.get(ConfigKeys.ECLBASE, DEFAULT_JOBNAME_FORMAT)
                ),
            ),
            eclbase_format_string=config_dict.get(
                ConfigKeys.ECLBASE,
                config_dict.get(ConfigKeys.JOBNAME, DEFAULT_ECLBASE_FORMAT),
            ),
            gen_kw_export_name=config_dict.get(
                ConfigKeys.GEN_KW_EXPORT_NAME, DEFAULT_GEN_KW_EXPORT_NAME
            ),
            time_map=time_map,
        )


def _replace_runpath_format(format_string: str) -> str:
    format_string = format_string.replace("%d", "<IENS>", 1)
    format_string = format_string.replace("%d", "<ITER>", 1)
    return format_string
