import pytest

from ert.config import ConfigValidationError, ErtConfig
from ert.plugins import ErtPluginContext


@pytest.mark.usefixtures("copy_poly_case")
def test_that_we_can_disable_a_parameter():
    with open("poly.ert", "a", encoding="utf-8") as fh:
        fh.writelines("GEN_KW DONT_UPDATE_KW template.txt kw.txt prior.txt\n")
        fh.writelines("LOAD_WORKFLOW config\n")
    with open("config", "w", encoding="utf-8") as fh:
        fh.writelines("DISABLE_PARAMETERS DONT_UPDATE_KW")
    with open("template.txt", "w", encoding="utf-8") as fh:
        fh.writelines("MY_KEYWORD <MY_KEYWORD>")
    with open("prior.txt", "w", encoding="utf-8") as fh:
        fh.writelines("MY_KEYWORD NORMAL 0 1")
    with (
        pytest.raises(ConfigValidationError, match="use the UPDATE:FALSE option"),
        ErtPluginContext() as ctx,
    ):
        ErtConfig.with_plugins(ctx).from_file("poly.ert")
