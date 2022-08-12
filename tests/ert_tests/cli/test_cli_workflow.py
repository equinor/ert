import os
from argparse import Namespace

from ert.shared.cli.workflow import execute_workflow
from ert.shared.plugins.plugin_manager import ErtPluginContext
from res.enkf import EnKFMain, ResConfig

from ...utils import SOURCE_DIR
from ..ert_utils import ErtTest, tmpdir


class WorkflowTest(ErtTest):
    @tmpdir(str(SOURCE_DIR / "test-data" / "local" / "poly_example"))
    def test_executing_workflow(self):
        with ErtPluginContext():
            with open("test_wf", "w") as wf_file:
                wf_file.write("EXPORT_RUNPATH")

            config_file = "poly.ert"
            with open(config_file, "a") as file:
                file.write("LOAD_WORKFLOW test_wf")

            rc = ResConfig(user_config_file=config_file)
            rc.convertToCReference(None)
            ert = EnKFMain(rc)
            args = Namespace(name="test_wf")
            execute_workflow(ert, args.name)
            assert os.path.isfile(".ert_runpath_list")
