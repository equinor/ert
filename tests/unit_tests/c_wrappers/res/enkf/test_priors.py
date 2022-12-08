import gc

import pytest

from ert._c_wrappers.enkf import EnKFMain, ResConfig


def test_adding_priors(poly_case):
    m = poly_case
    run_context = m.create_ensemble_experiment_run_context(
        active_mask=[True] * 10,
        iteration=0,
    )
    m.sample_prior(run_context.sim_fs, run_context.active_realizations)
    m.createRunPath(run_context)
    del m
    gc.collect()

    with open("coeff_priors", "a", encoding="utf-8") as f:
        f.write("COEFF_D UNIFORM 0 5\n")
    m = EnKFMain(ResConfig("poly.ert"))

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
