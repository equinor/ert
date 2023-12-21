import functools
import re
from functools import partial
from pathlib import Path

import gstools as gs
import numpy as np
import pytest
import scipy as sp
import xarray as xr
import xtgeo
from iterative_ensemble_smoother import steplength_exponential

from ert.analysis import (
    ErtAnalysisError,
    UpdateConfiguration,
    iterative_smoother_update,
    smoother_update,
)
from ert.analysis._es_update import UpdateSettings
from ert.analysis.configuration import UpdateStep
from ert.analysis.row_scaling import RowScaling
from ert.config import Field, GenDataConfig, GenKwConfig
from ert.config.analysis_module import ESSettings, IESSettings
from ert.field_utils import Shape
from ert.storage.realization_storage_state import RealizationStorageState


@pytest.fixture
def update_config():
    return UpdateConfiguration(
        update_steps=[
            UpdateStep(
                name="ALL_ACTIVE",
                observations=["OBSERVATION"],
                parameters=["PARAMETER"],
            )
        ]
    )


@pytest.fixture
def uniform_parameter():
    return GenKwConfig(
        name="PARAMETER",
        forward_init=False,
        template_file="",
        transfer_function_definitions=[
            "KEY1 UNIFORM 0 1",
        ],
        output_file="kw.txt",
    )


@pytest.fixture
def obs():
    return xr.Dataset(
        {
            "observations": (["report_step", "index"], [[1.0, 1.0, 1.0]]),
            "std": (["report_step", "index"], [[0.1, 1.0, 10.0]]),
        },
        coords={"index": [0, 1, 2], "report_step": [0]},
        attrs={"response": "RESPONSE"},
    )


def remove_timestamp_from_logfile(log_file: Path):
    with open(log_file, "r", encoding="utf-8") as fin:
        buf = fin.read()
    buf = re.sub(
        r"Time: [0-9]{4}\.[0-9]{2}\.[0-9]{2} [0-9]{2}\:[0-9]{2}\:[0-9]{2}", "Time:", buf
    )
    with open(log_file, "w", encoding="utf-8") as fout:
        fout.write(buf)


@pytest.mark.parametrize("misfit_preprocess", [True, False])
def test_update_report(
    snake_oil_case_storage, snake_oil_storage, snapshot, misfit_preprocess
):
    """
    Note that this is now a snapshot test, so there is no guarantee that the
    snapshots are correct, they are just documenting the current behavior.
    """
    ert_config = snake_oil_case_storage
    prior_ens = snake_oil_storage.get_ensemble_by_name("default_0")
    posterior_ens = snake_oil_storage.create_ensemble(
        prior_ens.experiment_id,
        ensemble_size=ert_config.model_config.num_realizations,
        iteration=1,
        name="new_ensemble",
        prior_ensemble=prior_ens,
    )
    smoother_update(
        prior_ens,
        posterior_ens,
        "id",
        UpdateConfiguration.global_update_step(
            list(ert_config.observations.keys()),
            ert_config.ensemble_config.parameters,
        ),
        UpdateSettings(misfit_preprocess=misfit_preprocess),
        ESSettings(ies_inversion=1),
        log_path=Path("update_log"),
    )
    log_file = Path(ert_config.analysis_config.log_path) / "id.txt"
    remove_timestamp_from_logfile(log_file)
    snapshot.assert_match(log_file.read_text("utf-8"), "update_log")


std_enkf_values = [
    0.4658755223614102,
    0.08294244626646294,
    -1.2728836885070545,
    -0.7044037773899394,
    0.0701040026601418,
    0.25463877762608783,
    -1.7638615728377676,
    1.0900234695729822,
    -1.2135225153906364,
    1.27516244886867,
]


@pytest.mark.parametrize(
    "module, expected_gen_kw, row_scaling",
    [
        (
            "IES_ENKF",
            [
                0.5167529669896218,
                -0.9178847938402281,
                -0.6299046429604261,
                -0.1632005925319205,
                0.0216488942750398,
                0.07464619425897459,
                -1.5587692532545538,
                0.22910522740018124,
                -0.7171489000139469,
                0.7287252249699406,
            ],
            False,
        ),
        (
            "STD_ENKF",
            [
                1.3040645145742686,
                -0.8162878122658299,
                -1.5484856041224397,
                -1.379896334985399,
                -0.510970027650022,
                0.5638868158813687,
                -2.7669280724377487,
                1.7160680670028017,
                -1.2603717378211836,
                1.2014197463741136,
            ],
            False,
        ),
        (
            "STD_ENKF",
            [
                0.7194682979730067,
                -0.5643616537018902,
                -1.341635690332394,
                -1.6888363123882548,
                -0.9922000342169071,
                0.6511460884255119,
                -2.5957226375270688,
                1.6899446147608206,
                -0.8679310950640513,
                1.2136685857887182,
            ],
            True,
        ),
    ],
)
def test_update_snapshot(
    snake_oil_case_storage,
    snake_oil_storage,
    module,
    expected_gen_kw,
    row_scaling,
):
    """
    Note that this is now a snapshot test, so there is no guarantee that the
    snapshots are correct, they are just documenting the current behavior.
    """
    ert_config = snake_oil_case_storage

    # Making sure that row scaling with a row scaling factor of 1.0
    # results in the same update as with ES.
    # Note: seed must be the same!
    if row_scaling:
        row_scaling = RowScaling()
        row_scaling.assign(10, lambda x: 1.0)
        update_step = UpdateStep(
            name="Row scaling only",
            observations=list(ert_config.observations.keys()),
            row_scaling_parameters=[("SNAKE_OIL_PARAM", row_scaling)],
        )
        update_configuration = UpdateConfiguration(update_steps=[update_step])
    else:
        update_configuration = UpdateConfiguration.global_update_step(
            list(ert_config.observations.keys()),
            list(ert_config.ensemble_config.parameters),
        )

    prior_ens = snake_oil_storage.get_ensemble_by_name("default_0")
    posterior_ens = snake_oil_storage.create_ensemble(
        prior_ens.experiment_id,
        ensemble_size=ert_config.model_config.num_realizations,
        iteration=1,
        name="posterior",
        prior_ensemble=prior_ens,
    )

    # Make sure we always have the same seed in updates
    rng = np.random.default_rng(42)

    if module == "IES_ENKF":
        # Step length defined as a callable on sies-iterations
        sies_step_length = functools.partial(steplength_exponential)

        # The sies-smoother is initially optional
        sies_smoother = None

        # The initial_mask equals ens_mask on first iteration
        initial_mask = prior_ens.get_realization_mask_from_state(
            [RealizationStorageState.HAS_DATA]
        )

        # Call an iteration of SIES algorithm. Producing snapshot and SIES obj
        iterative_smoother_update(
            prior_storage=prior_ens,
            posterior_storage=posterior_ens,
            sies_smoother=sies_smoother,
            run_id="id",
            update_config=update_configuration,
            update_settings=UpdateSettings(),
            analysis_config=IESSettings(ies_inversion=1),
            sies_step_length=sies_step_length,
            initial_mask=initial_mask,
            rng=rng,
        )
    else:
        smoother_update(
            prior_ens,
            posterior_ens,
            "id",
            update_configuration,
            UpdateSettings(),
            ESSettings(ies_inversion=1),
            rng=rng,
        )

    sim_gen_kw = list(
        prior_ens.load_parameters("SNAKE_OIL_PARAM", 0)["values"].values.flatten()
    )

    target_gen_kw = list(
        posterior_ens.load_parameters("SNAKE_OIL_PARAM", 0)["values"].values.flatten()
    )

    # Check that prior is not equal to posterior after updationg
    assert sim_gen_kw != target_gen_kw

    # Check that posterior is as expected
    assert target_gen_kw == pytest.approx(expected_gen_kw)


@pytest.mark.parametrize(
    "expected_target_gen_kw, update_step",
    [
        (
            [
                0.5895781800838542,
                -0.4369786388277017,
                -1.370782409107295,
                0.7564469588868706,
                0.21572672272162152,
                -0.24082711750101563,
                -1.1445220433012324,
                -1.03467093177391,
                -0.17607955213742074,
                0.02826184434039854,
            ],
            [
                {
                    "name": "update_step_LOCA",
                    "observations": ["WOPR_OP1_72"],
                    "parameters": [("SNAKE_OIL_PARAM", [1, 2])],
                }
            ],
        ),
        (
            [
                -4.47905516481858,
                -0.4369786388277017,
                1.1932696713609265,
                0.7564469588868706,
                0.21572672272162152,
                -0.24082711750101563,
                -1.1445220433012324,
                -1.03467093177391,
                -0.17607955213742074,
                0.02826184434039854,
            ],
            [
                {
                    "name": "update_step_LOCA",
                    "observations": ["WOPR_OP1_72"],
                    "parameters": [("SNAKE_OIL_PARAM", [1, 2])],
                },
                {
                    "name": "update_step_LOCA",
                    "observations": ["WOPR_OP1_108"],
                    "parameters": [("SNAKE_OIL_PARAM", [0, 2])],
                },
            ],
        ),
    ],
)
def test_localization(
    snake_oil_case_storage,
    snake_oil_storage,
    expected_target_gen_kw,
    update_step,
):
    """
    Note that this is now a snapshot test, so there is no guarantee that the
    snapshots are correct, they are just documenting the current behavior.
    """
    ert_config = snake_oil_case_storage

    # Row scaling with a scaling factor of 0.0 should result in no update,
    # which means that applying row scaling with a scaling factor of 0.0
    # should not change the snapshot.
    row_scaling = RowScaling()
    row_scaling.assign(10, lambda x: 0.0)
    for us in update_step:
        us["row_scaling_parameters"] = [("SNAKE_OIL_PARAM", row_scaling)]

    update_config = UpdateConfiguration(update_steps=update_step)

    prior_ens = snake_oil_storage.get_ensemble_by_name("default_0")

    posterior_ens = snake_oil_storage.create_ensemble(
        prior_ens.experiment_id,
        ensemble_size=ert_config.model_config.num_realizations,
        iteration=1,
        name="posterior",
        prior_ensemble=prior_ens,
    )
    smoother_update(
        prior_ens,
        posterior_ens,
        "an id",
        update_config,
        UpdateSettings(),
        ESSettings(ies_inversion=1),
        rng=np.random.default_rng(42),
    )

    sim_gen_kw = list(
        prior_ens.load_parameters("SNAKE_OIL_PARAM", 0)["values"].values.flatten()
    )

    target_gen_kw = list(
        posterior_ens.load_parameters("SNAKE_OIL_PARAM", 0)["values"].values.flatten()
    )

    # Test that the localized values has been updated
    assert sim_gen_kw[1:3] != target_gen_kw[1:3]

    # test that all the other values are left unchanged
    assert sim_gen_kw[3:] == target_gen_kw[3:]

    assert target_gen_kw == pytest.approx(expected_target_gen_kw)


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    "alpha, expected",
    [
        pytest.param(
            0.001,
            [],
            id="Low alpha, no active observations",
            marks=pytest.mark.xfail(raises=ErtAnalysisError, strict=True),
        ),
        (0.1, ["Deactivated, outlier", "Deactivated, outlier", "Active"]),
        (0.5, ["Deactivated, outlier", "Active", "Active"]),
        (1, ["Active", "Active", "Active"]),
    ],
)
def test_snapshot_alpha(
    alpha, expected, storage, uniform_parameter, update_config, obs
):
    """
    Note that this is now a snapshot test, so there is no guarantee that the
    snapshots are correct, they are just documenting the current behavior.
    """

    # alpha is a parameter used for outlier detection

    resp = GenDataConfig(name="RESPONSE")
    experiment = storage.create_experiment(
        parameters=[uniform_parameter],
        responses=[resp],
        observations={"OBSERVATION": obs},
    )
    prior_storage = storage.create_ensemble(
        experiment,
        ensemble_size=10,
        iteration=0,
        name="prior",
    )
    rng = np.random.default_rng(1234)
    for iens in range(prior_storage.ensemble_size):
        prior_storage.state_map[iens] = RealizationStorageState.HAS_DATA
        data = rng.uniform(0, 1)
        prior_storage.save_parameters(
            "PARAMETER",
            iens,
            xr.Dataset(
                {
                    "values": ("names", [data]),
                    "transformed_values": ("names", [data]),
                    "names": ["KEY_1"],
                }
            ),
        )
        data = rng.uniform(0.8, 1, 3)
        prior_storage.save_response(
            "RESPONSE",
            xr.Dataset(
                {"values": (["report_step", "index"], [data])},
                coords={"index": range(len(data)), "report_step": [0]},
            ),
            iens,
        )
    posterior_storage = storage.create_ensemble(
        prior_storage.experiment_id,
        ensemble_size=prior_storage.ensemble_size,
        iteration=1,
        name="posterior",
        prior_ensemble=prior_storage,
    )

    # Step length defined as a callable on sies-iterations
    sies_step_length = functools.partial(steplength_exponential)

    # The sies-smoother is initially optional
    sies_smoother = None

    # The initial_mask equals ens_mask on first iteration
    initial_mask = prior_storage.get_realization_mask_from_state(
        [RealizationStorageState.HAS_DATA]
    )

    result_snapshot, _ = iterative_smoother_update(
        prior_storage=prior_storage,
        posterior_storage=posterior_storage,
        sies_smoother=sies_smoother,
        run_id="id",
        update_config=update_config,
        update_settings=UpdateSettings(alpha=alpha),
        analysis_config=IESSettings(),
        sies_step_length=sies_step_length,
        initial_mask=initial_mask,
    )
    assert result_snapshot.alpha == alpha
    assert [
        obs.status for obs in result_snapshot.update_step_snapshots["ALL_ACTIVE"]
    ] == expected


def test_and_benchmark_adaptive_localization_with_fields(
    storage, tmp_path, monkeypatch, benchmark
):
    monkeypatch.chdir(tmp_path)

    rng = np.random.default_rng(42)

    num_grid_cells = 40
    num_parameters = num_grid_cells * num_grid_cells
    num_observations = 50
    num_ensemble = 25

    # Create a tridiagonal matrix that maps responses to parameters.
    # Being tridiagonal, it ensures that each response is influenced only by its neighboring parameters.
    diagonal = np.ones(min(num_parameters, num_observations))
    A = sp.sparse.diags(
        [diagonal, diagonal, diagonal],
        offsets=[-1, 0, 1],
        shape=(num_observations, num_parameters),
        dtype=float,
    ).toarray()

    # We add some noise that is insignificant compared to the
    # actual local structure in the forward model
    A = A + rng.standard_normal(size=A.shape) * 0.01

    def g(X):
        """Apply the forward model."""
        return A @ X

    # Initialize an ensemble representing the prior distribution of parameters using spatial random fields.
    model = gs.Exponential(dim=2, var=2, len_scale=8)
    fields = []
    seed = gs.random.MasterRNG(20170519)
    for _ in range(num_ensemble):
        srf = gs.SRF(model, seed=seed())
        field = srf.structured([np.arange(num_grid_cells), np.arange(num_grid_cells)])
        fields.append(field)
    X = np.vstack([field.flatten() for field in fields]).T

    Y = g(X)

    # Create observations by adding noise to a realization.
    observation_noise = rng.standard_normal(size=num_observations)
    observations = Y[:, 0] + observation_noise

    # Create necessary files and data sets to be able to update
    # the parameters using the ensemble smoother.
    shape = Shape(num_grid_cells, num_grid_cells, 1)
    grid = xtgeo.create_box_grid(dimension=(shape.nx, shape.ny, shape.nz))
    grid.to_file("MY_EGRID.EGRID", "egrid")

    resp = GenDataConfig(name="RESPONSE")
    obs = xr.Dataset(
        {
            "observations": (
                ["report_step", "index"],
                observations.reshape((1, num_observations)),
            ),
            "std": (
                ["report_step", "index"],
                observation_noise.reshape(1, num_observations),
            ),
        },
        coords={"report_step": [0], "index": np.arange(len(observations))},
        attrs={"response": "RESPONSE"},
    )

    param_group = "PARAM_FIELD"
    update_config = UpdateConfiguration(
        update_steps=[
            UpdateStep(
                name="ALL_ACTIVE",
                observations=["OBSERVATION"],
                parameters=[param_group],
            )
        ]
    )

    config = Field.from_config_list(
        "MY_EGRID.EGRID",
        shape,
        [
            param_group,
            param_group,
            "param.GRDECL",
            "INIT_FILES:param_%d.GRDECL",
            "FORWARD_INIT:False",
        ],
    )

    experiment = storage.create_experiment(
        parameters=[config],
        responses=[resp],
        observations={"OBSERVATION": obs},
    )

    prior = storage.create_ensemble(
        experiment,
        ensemble_size=num_ensemble,
        iteration=0,
        name="prior",
    )

    for iens in range(prior.ensemble_size):
        prior.state_map[iens] = RealizationStorageState.HAS_DATA
        prior.save_parameters(
            param_group,
            iens,
            xr.Dataset(
                {
                    "values": xr.DataArray(fields[iens], dims=("x", "y")),
                }
            ),
        )

        prior.save_response(
            "RESPONSE",
            xr.Dataset(
                {"values": (["report_step", "index"], [Y[:, iens]])},
                coords={"index": range(len(Y[:, iens])), "report_step": [0]},
            ),
            iens,
        )

    posterior_ens = storage.create_ensemble(
        prior.experiment_id,
        ensemble_size=prior.ensemble_size,
        iteration=1,
        name="posterior",
        prior_ensemble=prior,
    )

    smoother_update_run = partial(
        smoother_update,
        prior,
        posterior_ens,
        "id",
        update_config,
        UpdateSettings(),
        ESSettings(localization=True),
    )
    benchmark(smoother_update_run)

    prior_da = prior.load_parameters(param_group, range(num_ensemble))["values"]
    posterior_da = posterior_ens.load_parameters(param_group, range(num_ensemble))[
        "values"
    ]
    # Make sure some, but not all parameters were updated.
    assert not np.allclose(prior_da, posterior_da)
    # All parameters would be updated with a global update so this would fail.
    assert np.isclose(prior_da, posterior_da).sum() > 0


def test_update_only_using_subset_observations(
    snake_oil_case_storage, snake_oil_storage, snapshot
):
    """
    Note that this is now a snapshot test, so there is no guarantee that the
    snapshots are correct, they are just documenting the current behavior.
    """
    ert_config = snake_oil_case_storage
    update_config = UpdateConfiguration(
        update_steps=[
            {
                "name": "DISABLED_OBSERVATIONS",
                "observations": [
                    {"name": "FOPR", "index_list": [1]},
                    {"name": "WPR_DIFF_1"},
                ],
                "parameters": ert_config.ensemble_config.parameters,
            }
        ]
    )

    prior_ens = snake_oil_storage.get_ensemble_by_name("default_0")
    posterior_ens = snake_oil_storage.create_ensemble(
        prior_ens.experiment_id,
        ensemble_size=ert_config.model_config.num_realizations,
        iteration=1,
        name="new_ensemble",
        prior_ensemble=prior_ens,
    )
    smoother_update(
        prior_ens,
        posterior_ens,
        "id",
        update_config,
        UpdateSettings(),
        ESSettings(),
        log_path=Path(ert_config.analysis_config.log_path),
    )
    log_file = Path(ert_config.analysis_config.log_path) / "id.txt"
    remove_timestamp_from_logfile(log_file)
    snapshot.assert_match(log_file.read_text("utf-8"), "update_log")
