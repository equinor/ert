import pytest

from ert.config import ConfigValidationError, ErtConfig
from ert.plugins import ErtPluginContext


@pytest.mark.usefixtures("copy_poly_case")
def test_that_misfit_preprocessor_raises():
    with open("poly.ert", "a", encoding="utf-8") as fh:
        fh.writelines("LOAD_WORKFLOW config\n")
        fh.writelines("HOOK_WORKFLOW config PRE_FIRST_UPDATE\n")
    with open("config", "w", encoding="utf-8") as fh:
        fh.writelines("MISFIT_PREPROCESSOR")
    with (
        pytest.raises(
            ConfigValidationError,
            match="MISFIT_PREPROCESSOR is removed, use ANALYSIS_SET_VAR OBSERVATIONS",
        ),
        ErtPluginContext() as ctx,
    ):
        ErtConfig.with_plugins(ctx).from_file("poly.ert")


@pytest.mark.usefixtures("copy_poly_case")
def test_that_misfit_preprocessor_raises_with_config():
    with open("poly.ert", "a", encoding="utf-8") as fh:
        fh.writelines("LOAD_WORKFLOW config\n")
        fh.writelines("HOOK_WORKFLOW config PRE_FIRST_UPDATE\n")
    with open("config", "w", encoding="utf-8") as fh:
        fh.writelines("MISFIT_PREPROCESSOR my_config")
    with (
        pytest.raises(
            ConfigValidationError,
            match="Add multiple entries to set up multiple groups",
        ),
        ErtPluginContext() as ctx,
    ):
        ErtConfig.with_plugins(ctx).from_file("poly.ert")
