from ecl.util.util import BoolVector
from res.util.substitution_list import SubstitutionList
from ecl.util.test import TestAreaContext
from tests import ResTest
from tests.utils import tmpdir
from res.enkf import ErtRunContext
from res.enkf.enums import EnkfRunType
from res.enkf import EnkfFs
from res.util import PathFormat


class ErtRunContextTest(ResTest):
    @tmpdir()
    def test_create(self):
        with TestAreaContext("run_context"):
            sim_fs = EnkfFs.createFileSystem("sim_fs")
            target_fs = None

            mask = BoolVector(initial_size=100, default_value=True)
            mask[50] = False
            runpath_fmt = PathFormat("path/to/sim%d")
            subst_list = SubstitutionList()
            itr = 0
            jobname_fmt = "job%d"
            run_context1 = ErtRunContext(
                EnkfRunType.ENSEMBLE_EXPERIMENT,
                sim_fs,
                target_fs,
                mask,
                runpath_fmt,
                jobname_fmt,
                subst_list,
                itr,
            )
            run_id1 = run_context1.get_id()

            run_arg0 = run_context1[0]
            with self.assertRaises(ValueError):
                run_arg0.getQueueIndex()

            self.assertEqual(run_arg0.iter_id, itr)
            self.assertEqual(run_id1, run_arg0.get_run_id())

            run_context2 = ErtRunContext(
                EnkfRunType.ENSEMBLE_EXPERIMENT,
                sim_fs,
                target_fs,
                mask,
                runpath_fmt,
                jobname_fmt,
                subst_list,
                itr,
            )
            run_id2 = run_context2.get_id()

            self.assertFalse(run_id1 == run_id2)

            self.assertTrue(run_context1.is_active(49))
            self.assertFalse(run_context1.is_active(50))
