import gc
import os
import shutil

import pytest

from ert._c_wrappers.enkf import EnKFMain, ResConfig, RunContext


def test_adding_priors(tmpdir, source_root):
    shutil.copytree(
        os.path.join(source_root, "test-data", "poly_example"),
        os.path.join(str(tmpdir), "poly_example"),
    )
    with tmpdir.as_cwd():
        rc = ResConfig("poly_example/poly.ert")
        m = EnKFMain(rc)
        run_context = m.create_ensemble_experiment_run_context(
            active_mask=[True] * 10,
            iteration=0,
        )
        m.sample_prior(run_context.sim_fs, run_context.active_realizations)
        m.createRunPath(run_context)
        del m
        gc.collect()

        with open("poly_example/coeff_priors", "a") as f:
            f.write("COEFF_D UNIFORM 0 5\n")
        rc = ResConfig("poly_example/poly.ert")
        m = EnKFMain(rc)

        run_context = m.create_ensemble_experiment_run_context(
            active_mask=[True] * 10,
            iteration=0,
        )
        with pytest.raises(
            ValueError,
            match="The configuration of GEN_KW "
            "parameter COEFFS is of size 4, expected 3",
        ):
            m.createRunPath(run_context)
