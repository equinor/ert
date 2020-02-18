import unittest, os
import itertools

from ecl.util.test import TestAreaContext
from tests import ResTest
from tests.utils import tmpdir
from res.test import ErtTestContext

from ecl.util.util import BoolVector

from res.enkf import ResConfig, EnKFMain, EnkfFs, ErtRunContext
from res.enkf.enums import EnKFFSType, EnkfRunType
from res.util import PathFormat


def render_dynamic_values(s, itr, iens, geo_id):
    dynamic_magic_strings = {
        "<GEO_ID>": geo_id,
        "<ITER>": itr,
        "<IENS>": iens,
    }
    for key, val in dynamic_magic_strings.items():
        s = s.replace(key, str(val))

    return s


class RunpathListDumpTest(ResTest):

    def setUp(self):
        self.config_rel_path = "local/snake_oil_no_data/snake_oil_GEO_ID.ert"
        self.config_path = self.createTestPath(self.config_rel_path)


    def _verify_runpath_rendering(self, itr):
        with ErtTestContext("add_all_runpath_dump", model_config=self.config_path) as ctx:
            res = ctx.getErt()
            fs_manager = res.getEnkfFsManager()
            sim_fs = fs_manager.getFileSystem("sim_fs")

            num_realizations = 25
            mask = BoolVector(initial_size=num_realizations, default_value=True)
            mask[13] = False

            runpath_fmt = "simulations/<GEO_ID>/realisation-%d/iter-%d/magic-real-<IENS>/magic-iter-<ITER>"
            jobname_fmt = "SNAKE_OIL_%d"

            subst_list = res.resConfig().subst_config.subst_list
            run_context = ErtRunContext.ensemble_experiment(sim_fs, mask, PathFormat(runpath_fmt), jobname_fmt, subst_list, itr)

            res.initRun(run_context)

            for i, run_arg in enumerate(run_context):
                if mask[i]:
                    run_arg.geo_id = 10*i

            res.createRunpath(run_context)

            for i, run_arg in enumerate(run_context):
                if not mask[i]:
                    continue

                self.assertTrue(os.path.isdir("simulations/%d" % run_arg.geo_id))

            runpath_list_path = ".ert_runpath_list"
            self.assertTrue(os.path.isfile(runpath_list_path))

            exp_runpaths = [
                render_dynamic_values(runpath_fmt, itr, iens, run_arg.geo_id) % (iens, itr)
                for iens, run_arg in enumerate(run_context) if mask[iens]
            ]
            exp_runpaths = map(os.path.realpath, exp_runpaths)

            with open(runpath_list_path, 'r') as f:
                dumped_runpaths = list(zip(*[line.split() for line in f.readlines()]))[1]

            self.assertEqual(list(exp_runpaths), list(dumped_runpaths))


    @tmpdir()
    def test_add_all(self):
        test_base = [0, 1, 2, 17]
        for itr in test_base:
            self._verify_runpath_rendering(itr)
