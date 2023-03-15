import gc

import pytest

from ert._c_wrappers.enkf import EnKFMain, ErtConfig


def test_adding_priors(poly_case, storage):
    m = poly_case
    experiment_id = storage.create_experiment(
        parameters=m.ensembleConfig().parameter_configuration
    )
    prior_ensemble = storage.create_ensemble(
        experiment_id, name="prior", ensemble_size=10
    )
    prior = m.ensemble_context(
        prior_ensemble,
        [True] * 10,
        iteration=0,
    )
    m.sample_prior(prior.sim_fs, prior.active_realizations)
    m.createRunPath(prior)
    del m
    gc.collect()

    with open("coeff_priors", "a", encoding="utf-8") as f:
        f.write("COEFF_D UNIFORM 0 5\n")
    m = EnKFMain(ErtConfig.from_file("poly.ert"))

    prior = m.ensemble_context(
        prior.sim_fs,
        [True] * 10,
        iteration=0,
    )
    with pytest.raises(
        ValueError,
        match="The configuration of GEN_KW "
        "parameter COEFFS is of size 4, expected 3",
    ):
        m.createRunPath(prior)
