import os
from pathlib import Path

import pytest


def test_field_basics(snake_oil_field_example):
    ert = snake_oil_field_example
    ens_config = ert.ensembleConfig()
    fc = ens_config["PERMX"]

    assert (fc.nx, fc.ny, fc.nz) == (10, 10, 5)
    assert fc.truncation_min is None
    assert fc.truncation_max is None
    assert fc.input_transformation is None
    assert fc.output_transformation is None


def test_field_export(snake_oil_field_example, storage):
    ert = snake_oil_field_example
    experiment_id = storage.create_experiment(
        parameters=ert.ensembleConfig().parameter_configuration
    )
    prior_ensemble = storage.create_ensemble(
        experiment_id, name="prior", ensemble_size=5
    )

    prior = ert.ensemble_context(prior_ensemble, [True, False, False, True, True], 0)
    ert.sample_prior(prior.sim_fs, prior.active_realizations)
    ens_config = ert.ensembleConfig()
    config_node = ens_config["PERMX"]

    for real in [0, 3, 4]:
        config_node.write_to_runpath(
            Path(f"export/with/path/{real}"), real, prior_ensemble
        )
        assert os.path.isfile(f"export/with/path/{real}/permx.grdecl")
        assert os.path.getsize(f"export/with/path/{real}/permx.grdecl") > 0
    for real in [1, 2]:
        with pytest.raises(
            KeyError, match=f"No dataset 'PERMX' in storage for realization {real}"
        ):
            config_node.write_to_runpath(
                Path(f"export/with/path/{real}"), real, prior_ensemble
            )
        assert not os.path.isfile(f"export/with/path/{real}/permx.grdecl")
