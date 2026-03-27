import pytest

from ert.config import ConfigValidationError, ConfigWarning, ErtConfig
from ert.plugins import get_site_plugins


@pytest.mark.usefixtures("copy_poly_case")
def test_that_removed_job_disable_parameters_gives_warning():
    # Warning is given on LOAD_WORKFLOW.
    # Since LOAD_WORKFLOW failed error is raised on HOOK_WORKFLOW
    with open("poly.ert", "a", encoding="utf-8") as fh:
        fh.writelines("GEN_KW DONT_UPDATE_KW template.txt kw.txt prior.txt\n")
        fh.writelines("LOAD_WORKFLOW config\n")
        fh.writelines("HOOK_WORKFLOW config PRE_SIMULATION\n")
    with open("config", "w", encoding="utf-8") as fh:
        fh.writelines("DISABLE_PARAMETERS DONT_UPDATE_KW")
    with open("template.txt", "w", encoding="utf-8") as fh:
        fh.writelines("MY_KEYWORD <MY_KEYWORD>")
    with open("prior.txt", "w", encoding="utf-8") as fh:
        fh.writelines("MY_KEYWORD NORMAL 0 1")
    with (
        pytest.warns(ConfigWarning, match="use the UPDATE:FALSE option"),
        pytest.raises(
            ConfigValidationError,
            match="Cannot setup hook for non-existing job name",
        ),
    ):
        ErtConfig.with_plugins(runtime_plugins=get_site_plugins()).from_file("poly.ert")
