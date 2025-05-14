from __future__ import annotations

import contextlib
import logging
import os.path
import shutil
from pathlib import Path
from typing import no_type_check

from pydantic import BaseModel, field_validator

from ert.shared.status.utils import byte_with_unit, get_mount_directory

from .parsing import (
    ConfigDict,
    ConfigKeys,
    ConfigValidationError,
    ConfigWarning,
)

logger = logging.getLogger(__name__)


DEFAULT_RUNPATH = "simulations/realization-<IENS>/iter-<ITER>"
DEFAULT_GEN_KW_EXPORT_NAME = "parameters"
DEFAULT_JOBNAME_FORMAT = "<CONFIG_FILE>-<IENS>"
DEFAULT_ECLBASE_FORMAT = "ECLBASE<IENS>"

FULL_DISK_PERCENTAGE_THRESHOLD = 0.97
MINIMUM_BYTES_LEFT_ON_DISK_THRESHOLD = 200 * 1000**3  # 200 GB
# We give warning if free disk space is less than MINIMUM_BYTES_LEFT_ON_DISK_THRESHOLD
# and used space in percentage is greater than FULL_DISK_PERCENTAGE_THRESHOLD


class ModelConfig(BaseModel):
    num_realizations: int = 1
    runpath_format_string: str = DEFAULT_RUNPATH
    jobname_format_string: str = DEFAULT_JOBNAME_FORMAT
    eclbase_format_string: str = DEFAULT_ECLBASE_FORMAT
    gen_kw_export_name: str = DEFAULT_GEN_KW_EXPORT_NAME

    @field_validator("runpath_format_string", mode="before")
    @classmethod
    def validate_runpath(cls, runpath_format_string: str) -> str:
        if "%d" in runpath_format_string and any(
            x in runpath_format_string for x in ["<ITER>", "<IENS>"]
        ):
            raise ConfigValidationError(
                "RUNPATH cannot combine deprecated and new style placeholders: "
                f"`{runpath_format_string}`. Valid example `{DEFAULT_RUNPATH}`"
            )
        # do not allow multiple occurrences
        for kw in ["<ITER>", "<IENS>"]:
            if runpath_format_string.count(kw) > 1:
                raise ConfigValidationError(
                    f"RUNPATH cannot contain multiple {kw} placeholders: "
                    f"`{runpath_format_string}`. Valid example `{DEFAULT_RUNPATH}`"
                )
        # do not allow too many placeholders
        if runpath_format_string.count("%d") > 2:
            raise ConfigValidationError(
                "RUNPATH cannot contain more than two value placeholders: "
                f"`{runpath_format_string}`. Valid example `{DEFAULT_RUNPATH}`"
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
            mount_dir = get_mount_directory(Path(runpath_format_string))
            total_space, used_space, free_space = shutil.disk_usage(mount_dir)
            percentage_used = used_space / total_space
            if (
                percentage_used > FULL_DISK_PERCENTAGE_THRESHOLD
                and free_space < MINIMUM_BYTES_LEFT_ON_DISK_THRESHOLD
            ):
                msg = (
                    f"Low disk space: {byte_with_unit(free_space)} "
                    f"free on {mount_dir!s}. Consider freeing up some "
                    "space to ensure successful simulation runs."
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
        return cls(
            num_realizations=config_dict.get(ConfigKeys.NUM_REALIZATIONS, 1),
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
        )


def _replace_runpath_format(format_string: str) -> str:
    format_string = format_string.replace("%d", "<IENS>", 1)
    format_string = format_string.replace("%d", "<ITER>", 1)
    return format_string
