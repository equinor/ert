from pathlib import Path

import pytest

from ert.config import ConfigValidationError, ConfigWarning, ErtConfig
from ert.plugins import get_site_plugins


@pytest.mark.usefixtures("copy_poly_case")
def test_that_misfit_preprocessor_raises():
    # Warning is given on LOAD_WORKFLOW.
    # Since LOAD_WORKFLOW failed error is raised on HOOK_WORKFLOW
    with Path("poly.ert").open("a", encoding="utf-8") as fh:
        fh.writelines("LOAD_WORKFLOW config\n")
        fh.writelines("HOOK_WORKFLOW config PRE_FIRST_UPDATE\n")
    with Path("config").open("w", encoding="utf-8") as fh:
        fh.writelines("MISFIT_PREPROCESSOR")
    with (
        pytest.warns(
            ConfigWarning,
            match=(
                r"Encountered the following error\(s\) while reading workflow "
                r"'config'. It will not be loaded:.*MISFIT_PREPROCESSOR is removed, "
                r"use ANALYSIS_SET_VAR OBSERVATIONS"
            ),
        ),
        pytest.raises(
            ConfigValidationError,
            match="Cannot setup hook for non-existing job name",
        ),
    ):
        ErtConfig.with_plugins(get_site_plugins()).from_file("poly.ert")


@pytest.mark.usefixtures("copy_poly_case")
def test_that_misfit_preprocessor_raises_with_config():
    # Warning is given on LOAD_WORKFLOW.
    # Since LOAD_WORKFLOW failed error is raised on HOOK_WORKFLOW
    with Path("poly.ert").open("a", encoding="utf-8") as fh:
        fh.writelines("LOAD_WORKFLOW config\n")
        fh.writelines("HOOK_WORKFLOW config PRE_FIRST_UPDATE\n")
    with Path("config").open("w", encoding="utf-8") as fh:
        fh.writelines("MISFIT_PREPROCESSOR my_config")
    with (
        pytest.warns(
            ConfigWarning,
            match=(
                r"(?s)Encountered the following error\(s\) while reading workflow "
                r"'config'. It will not be loaded:.*Add multiple entries to set up "
                r"multiple groups"
            ),
        ),
        pytest.raises(
            ConfigValidationError,
            match="Cannot setup hook for non-existing job name",
        ),
    ):
        ErtConfig.with_plugins(get_site_plugins()).from_file("poly.ert")
