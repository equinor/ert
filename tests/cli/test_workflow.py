import os
from argparse import Namespace

from ert_shared import ERT
from ert_shared.cli.notifier import ErtCliNotifier
from ert_shared.cli.workflow import execute_workflow
from res.enkf import EnKFMain, ResConfig
from tests import ErtTest
from tests.utils import SOURCE_DIR, tmpdir


class WorkflowTest(ErtTest):
    @tmpdir(os.path.join(SOURCE_DIR, "test-data/local/poly_example"))
    def test_executing_workflow(self):
        with open("test_wf", "w") as wf_file:
            wf_file.write("EXPORT_RUNPATH")

        config_file = "poly.ert"
        with open(config_file, "a") as file:
            file.write("LOAD_WORKFLOW test_wf")

        rc = ResConfig(user_config_file=config_file)
        rc.convertToCReference(None)
        ert = EnKFMain(rc)
        notifier = ErtCliNotifier(ert, config_file)
        ERT.adapt(notifier)
        args = Namespace(name="test_wf")
        execute_workflow(args.name)
        assert os.path.isfile(".ert_runpath_list")
