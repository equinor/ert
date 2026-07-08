from pathlib import Path
from textwrap import dedent

import numpy as np
import polars as pl
import pytest
import xarray as xr
from pandas import DataFrame, ExcelWriter

from ert.config import (
    DesignMatrix,
    ErtConfig,
    GenKwConfig,
)
from ert.config.design_matrix import DESIGN_MATRIX_GROUP
from ert.sample_prior import sample_prior
from ert.storage import (
    LocalEnsemble,
    open_storage,
)


def test_that_local_ensemble_save_parameter_raises_value_error_given_xr_array_dataset(
    tmp_path,
):
    storage = open_storage(tmp_path, mode="w")
    experiment = storage.create_experiment()
    local_ensemble = storage.create_ensemble(
        experiment, ensemble_size=1, iteration=0, name="prior"
    )

    expected_error_msg = (
        r"Dataset must be either an xarray Dataset or polars Dataframe, was 'ndarray'"
    )
    with pytest.raises(AssertionError, match=expected_error_msg):
        local_ensemble.save_parameters(dataset=np.array([]), group="", realization=None)


def test_that_saving_arbitrary_parameter_dataframe_fails(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        uniform_parameter = GenKwConfig(
            name="KEY_1",
            distribution={"name": "uniform", "min": 0, "max": 1},
        )
        experiment = storage.create_experiment(
            experiment_config={
                "parameter_configuration": [uniform_parameter.model_dump(mode="json")]
            }
        )
        prior = storage.create_ensemble(
            experiment, ensemble_size=1, iteration=0, name="prior"
        )

        with pytest.raises(
            ValueError,
            match=r"Parameters dataframe is empty.",
        ):
            prior.save_parameters(pl.DataFrame())

        with pytest.raises(
            KeyError,
            match="Columns KEY_2, KEY_3 not in experiment parameters",
        ):
            prior.save_parameters(
                pl.DataFrame(
                    {
                        "realization": [0],
                        "KEY_2": [1.0],
                        "KEY_3": [1.0],
                    }
                )
            )

        with pytest.raises(
            KeyError,
            match=(
                "DataFrame must contain a 'realization' column for"
                " saving scalar parameters"
            ),
        ):
            prior.save_parameters(
                pl.DataFrame(
                    {
                        "KEY_1": [1.0],
                    }
                )
            )


def test_that_saving_empty_parameters_fails_nicely(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment()
        prior = storage.create_ensemble(
            experiment, ensemble_size=1, iteration=0, name="prior"
        )

        # Test for entirely empty dataset
        with pytest.raises(
            ValueError,
            match=(
                "Dataset for parameter group 'PARAMETER' "
                "must contain a 'values' variable"
            ),
        ):
            prior.save_parameters(xr.Dataset(), "PARAMETER", 0)

        # Test for dataset with 'values' and 'transformed_values' but no actual data
        empty_data = xr.Dataset(
            {
                "values": ("names", np.array([], dtype=float)),
                "names": (["names"], np.array([], dtype=str)),
            }
        )
        with pytest.raises(
            ValueError,
            match=(
                "Parameters PARAMETER are empty\\. "
                "Cannot proceed with saving to storage\\."
            ),
        ):
            prior.save_parameters(empty_data, "PARAMETER", 0)


def test_that_loading_parameter_via_response_api_fails(tmp_path):
    uniform_parameter = GenKwConfig(
        name="KEY_1",
        distribution={"name": "uniform", "min": 0, "max": 1},
    )
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "parameter_configuration": [uniform_parameter.model_dump(mode="json")]
            },
        )
        prior = storage.create_ensemble(
            experiment,
            ensemble_size=1,
            iteration=0,
            name="prior",
        )

        prior.save_parameters(
            pl.DataFrame(
                {
                    "realization": [0],
                    "KEY_1": [1.0],
                }
            )
        )
        with pytest.raises(ValueError, match="KEY_1 is not a response"):
            prior.load_responses("KEY_1", (0,))


def test_that_load_parameters_throws_exception(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment()
        ensemble = storage.create_ensemble(experiment, name="foo", ensemble_size=1)

        with pytest.raises(expected_exception=KeyError):
            ensemble.load_parameters("I_DONT_EXIST", 1)


@pytest.mark.parametrize(
    ("reals", "expect_error"),
    [
        pytest.param(
            list(range(10)),
            False,
            id="correct_active_realizations",
        ),
        pytest.param([10, 11], True, id="incorrect_active_realizations"),
    ],
)
def test_sample_parameter_with_design_matrix(tmp_path, reals, expect_error):
    design_path = tmp_path / "design_matrix.xlsx"
    ensemble_size = 10
    a_values = np.random.default_rng().uniform(-5, 5, 10)
    b_values = np.random.default_rng().uniform(-5, 5, 10)
    c_values = np.random.default_rng().uniform(-5, 5, 10)
    design_matrix_df = DataFrame({"a": a_values, "b": b_values, "c": c_values})
    with ExcelWriter(design_path) as xl_write:
        design_matrix_df.to_excel(xl_write, index=False, sheet_name="DesignSheet")
        DataFrame().to_excel(
            xl_write, index=False, sheet_name="DefaultSheet", header=False
        )
    design_matrix = DesignMatrix(design_path, "DesignSheet", "DefaultSheet")
    with open_storage(tmp_path / "storage", mode="w") as storage:
        experiment_id = storage.create_experiment(
            experiment_config={
                "parameter_configuration": [
                    pc.model_dump(mode="json")
                    for pc in design_matrix.parameter_configurations
                ]
            }
        )
        ensemble = storage.create_ensemble(
            experiment_id, name="default", ensemble_size=ensemble_size
        )
        if expect_error:
            with pytest.raises(KeyError):
                sample_prior(
                    ensemble,
                    reals,
                    random_seed=123,
                    num_realizations=ensemble_size,
                    parameters=[
                        param.name for param in design_matrix.parameter_configurations
                    ],
                    design_matrix_df=design_matrix.design_matrix_df,
                )
        else:
            sample_prior(
                ensemble,
                reals,
                random_seed=123,
                num_realizations=ensemble_size,
                parameters=[
                    param.name for param in design_matrix.parameter_configurations
                ],
                design_matrix_df=design_matrix.design_matrix_df,
            )
            params = ensemble.load_parameters(DESIGN_MATRIX_GROUP).drop("realization")
            assert isinstance(params, pl.DataFrame)
            assert params.columns == ["a", "b", "c"]
            np.testing.assert_array_almost_equal(params["a"].to_list(), a_values)
            np.testing.assert_array_almost_equal(params["b"].to_list(), b_values)
            np.testing.assert_array_almost_equal(params["c"].to_list(), c_values)


def test_load_gen_kw_not_sorted(storage, tmpdir, snapshot):
    """
    This test checks two things, loading multiple parameters and
    loading log parameters.
    """
    with tmpdir.as_cwd():
        config = dedent(
            """
        NUM_REALIZATIONS 10
        GEN_KW PARAM_2 template.txt kw.txt prior2.txt
        GEN_KW PARAM_1 template.txt kw.txt prior1.txt
        RANDOM_SEED 1234
        """
        )
        with Path("config.ert").open(mode="w", encoding="utf-8") as fh:
            fh.writelines(config)
        with Path("template.txt").open(mode="w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD <MY_KEYWORD>")
        with Path("prior1.txt").open(mode="w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD1 LOGUNIF 0.1 1")
        with Path("prior2.txt").open(mode="w", encoding="utf-8") as fh:
            fh.writelines("MY_KEYWORD2 LOGUNIF 0.1 1")

        ert_config = ErtConfig.from_file("config.ert")

        experiment_id = storage.create_experiment(
            experiment_config={
                "parameter_configuration": [
                    cfg.model_dump(mode="json")
                    for cfg in ert_config.ensemble_config.parameter_configuration
                ]
            }
        )
        ensemble_size = 10
        ensemble = storage.create_ensemble(
            experiment_id, name="default", ensemble_size=ensemble_size
        )

        sample_prior(
            ensemble,
            range(ensemble_size),
            random_seed=1234,
            num_realizations=ensemble_size,
        )
        data = ensemble.load_scalars()
        snapshot.assert_match(data.write_csv(float_precision=12), "gen_kw_unsorted")


def test_gen_kw_collector(snake_oil_default_storage, snapshot):
    data: pl.DataFrame = snake_oil_default_storage.load_scalars()
    snapshot.assert_match(data.write_csv(float_precision=6), "gen_kw_collector.csv")

    with pytest.raises(KeyError):
        # realization 60:
        _ = data.to_dict()[60]

    realization_index = 3
    data: pl.DataFrame = snake_oil_default_storage.load_scalars(
        realizations=[realization_index],
    )
    snapshot.assert_match(data.write_csv(float_precision=6), "gen_kw_collector_3.csv")

    non_existing_realization_index = 150
    with pytest.raises(IndexError):
        data: pl.DataFrame = snake_oil_default_storage.load_scalars(
            realizations=[non_existing_realization_index],
        )


def test_that_multiple_save_parameters_numpy_calls_overwrite_previous_values(tmp_path):
    """This test is to ensure that updating parameters in local ensemble does not
    raise duplicate column errors when saving with numpy arrays multiple times, and
    instead overwrites the previous values as expected.
    """
    writer = open_storage(tmp_path, mode="w")
    gen_kw_parameter = GenKwConfig(
        name="some_param", distribution={"name": "normal", "mean": 10, "std": 0.1}
    )
    exp = writer.create_experiment(
        experiment_config={
            "parameter_configuration": [gen_kw_parameter.model_dump(mode="json")],
        }
    )
    num_reals = 5
    parameter_data_0 = np.array([0.0] * num_reals)
    parameter_data_1 = np.array([1.1] * num_reals)
    parameter_data_2 = np.array([2.2] * num_reals)
    ens: LocalEnsemble = exp.create_ensemble(ensemble_size=num_reals, name="uniq_ens")
    iens_active_index = np.array(range(num_reals))
    ens.save_parameters_numpy(
        parameter_data_0, gen_kw_parameter.name, iens_active_index
    )
    ens.save_parameters_numpy(
        parameter_data_1, gen_kw_parameter.name, iens_active_index
    )
    ens.save_parameters_numpy(
        parameter_data_2, gen_kw_parameter.name, iens_active_index
    )
    saved_ens = ens.load_scalar_keys([gen_kw_parameter.name]).to_dict()[
        gen_kw_parameter.name
    ]
    assert saved_ens.to_list() == [2.2 for _ in range(num_reals)]
