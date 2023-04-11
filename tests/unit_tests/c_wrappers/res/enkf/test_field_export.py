import os

import pytest
from ecl.grid import EclGrid


def test_field_basics(snake_oil_field_example):
    ert = snake_oil_field_example
    ens_config = ert.ensembleConfig()
    fc = ens_config["PERMX"]
    grid = EclGrid(fc.grid_file)

    assert (fc.nx, fc.ny, fc.nz) == (10, 10, 5)
    assert (grid.nx, grid.ny, grid.nz) == (10, 10, 5)
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
    ert.createRunPath(prior)
    ens_config = ert.ensembleConfig()
    config_node = ens_config["PERMX"]

    fs = prior_ensemble
    fs.export_field(
        config_node.name,
        0,
        "export/with/path/PERMX_0.grdecl",
        fformat="grdecl",
    )
    assert os.path.isfile("export/with/path/PERMX_0.grdecl")
    assert os.path.getsize("export/with/path/PERMX_0.grdecl") > 0

    with pytest.raises(
        KeyError, match="Unable to load FIELD for key: PERMX, realization: 1"
    ):
        fs.export_field(
            config_node.name,
            1,
            "export/with/path/PERMX_1.grdecl",
            fformat="grdecl",
        )
    assert not os.path.isfile("export/with/path/PERMX_1.grdecl")

    with pytest.raises(
        KeyError, match="Unable to load FIELD for key: PERMX, realization: 2"
    ):
        fs.export_field(
            config_node.name,
            2,
            "export/with/path/PERMX_2.grdecl",
            fformat="grdecl",
        )
    assert not os.path.isfile("export/with/path/PERMX_2.grdecl")

    fs.export_field(
        config_node.name,
        3,
        "export/with/path/PERMX_3.grdecl",
        fformat="grdecl",
    )
    assert os.path.isfile("export/with/path/PERMX_3.grdecl")
    assert os.path.getsize("export/with/path/PERMX_3.grdecl") > 0

    fs.export_field(
        config_node.name,
        4,
        "export/with/path/PERMX_4.grdecl",
        fformat="grdecl",
    )
    assert os.path.isfile("export/with/path/PERMX_4.grdecl")
    assert os.path.getsize("export/with/path/PERMX_4.grdecl") > 0
