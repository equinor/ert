import os

from res.test import ErtTestContext

from ...libres_utils import ResTest, tmpdir


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
        self.config_rel_path = "local/snake_oil/snake_oil.ert"
        self.config_path = self.createTestPath(self.config_rel_path)

    def _verify_runpath_rendering(self, itr):
        with ErtTestContext(model_config=self.config_path) as ctx:
            res = ctx.getErt()
            fs_manager = res.getEnkfFsManager()
            sim_fs = fs_manager.getFileSystem("sim_fs")

            num_realizations = 25
            mask = [True] * num_realizations
            mask[13] = False
            runpath_fmt = (
                "simulations/<GEO_ID>/realization-%d/iter-%d/"
                "magic-real-<IENS>/magic-iter-<ITER>"
            )
            jobname_fmt = "SNAKE_OIL_%d"
            res.runpaths._runpath_format = runpath_fmt
            res.runpaths._job_name_format = jobname_fmt

            run_context = res.create_ensemble_experiment_run_context(
                source_filesystem=sim_fs,
                active_mask=mask,
                iteration=itr,
            )

            res.initRun(run_context)

            for i, run_arg in enumerate(run_context):
                if mask[i]:
                    res.set_geo_id(str(10 * i), i, itr)

            res.createRunPath(run_context)

            for i, run_arg in enumerate(run_context):
                if not mask[i]:
                    continue

                self.assertTrue(os.path.isdir(f"simulations/{10*i}"))

            runpath_list_path = ".ert_runpath_list"
            self.assertTrue(os.path.isfile(runpath_list_path))

            exp_runpaths = [
                render_dynamic_values(runpath_fmt, itr, iens, str(10 * iens))
                % (iens, itr)
                for iens, run_arg in enumerate(run_context)
                if mask[iens]
            ]
            exp_runpaths = map(os.path.realpath, exp_runpaths)

            with open(runpath_list_path, "r") as f:
                dumped_runpaths = list(zip(*[line.split() for line in f.readlines()]))[
                    1
                ]

            self.assertEqual(list(exp_runpaths), list(dumped_runpaths))

    @tmpdir()
    def test_add_all(self):
        test_base = [0, 1, 2, 17]
        for itr in test_base:
            self._verify_runpath_rendering(itr)
