import unittest, os

from ecl.util.test import TestAreaContext
from tests import ResTest
from res.test import ErtTestContext

from ecl.util.util import BoolVector

from res.enkf import ResConfig, EnKFMain, EnkfFs, ErtRunContext
from res.enkf.enums import EnKFFSType, EnkfRunType
from res.util import PathFormat

class RunpathListDumpTest(ResTest):

    def setUp(self):
        self.config_rel_path = "local/snake_oil_no_data/snake_oil_GEO_ID.ert"
        self.config_path = self.createTestPath(self.config_rel_path)

    def test_add_all(self):
        with ErtTestContext("add_all_runpath_dump", model_config=self.config_path) as ctx:
            res = ctx.getErt()
            fs_manager = res.getEnkfFsManager()
            sim_fs = fs_manager.getFileSystem("sim_fs")

            num_realizations = 25
            mask = BoolVector(initial_size=num_realizations, default_value=True)
            mask[13] = False

            runpath_fmt = "simulations/<GEO_ID>/realisation-%d/iter-%d"
            jobname_fmt = "SNAKE_OIL_%d"

            itr = 0
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
                             runpath_fmt.replace("<GEO_ID>", str(run_arg.geo_id)) % (iens, itr)
                             for iens, run_arg in enumerate(run_context) if mask[iens]
                           ]
            exp_runpaths = map(os.path.realpath, exp_runpaths)

            with open(runpath_list_path, 'r') as f:
                dumped_runpaths = list(zip(*[line.split() for line in f.readlines()]))[1]

            self.assertEqual(list(exp_runpaths), list(dumped_runpaths))


    def test_add_one_by_one(self):
        with ErtTestContext("add_one_by_one_runpath_dump", model_config=self.config_path) as ctx:
            res = ctx.getErt()
            fs_manager = res.getEnkfFsManager()
            sim_fs = fs_manager.getFileSystem("sim_fs")

            num_realizations = 25
            mask = BoolVector(initial_size=num_realizations, default_value=True)
            mask[13] = False

            runpath_fmt = "simulations/<GEO_ID>/realisation-%d/iter-%d"
            jobname_fmt = "SNAKE_OIL_%d"

            itr = 0
            subst_list = res.resConfig().subst_config.subst_list
            run_context = ErtRunContext.ensemble_experiment(sim_fs, mask, PathFormat(runpath_fmt), jobname_fmt, subst_list, itr)

            res.initRun(run_context)

            for i, run_arg in enumerate(run_context):
                if mask[i]:
                    run_arg.geo_id = 10*i
                    res.createRunpath(run_context, i)

            for i, run_arg in enumerate(run_context):
                if not mask[i]:
                    continue

                self.assertTrue(os.path.isdir("simulations/%d" % run_arg.geo_id))

            runpath_list_path = ".ert_runpath_list"
            self.assertTrue(os.path.isfile(runpath_list_path))

            exp_runpaths = [
                             runpath_fmt.replace("<GEO_ID>", str(run_arg.geo_id)) % (iens, itr)
                             for iens, run_arg in enumerate(run_context) if mask[iens]
                           ]
            exp_runpaths = map(os.path.realpath, exp_runpaths)

            with open(runpath_list_path, 'r') as f:
                dumped_runpaths = list(zip(*[line.split() for line in f.readlines()]))[1]

            self.assertEqual(list(exp_runpaths), list(dumped_runpaths))
