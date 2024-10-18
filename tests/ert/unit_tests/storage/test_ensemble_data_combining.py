import datetime
import random
from typing import List

import numpy as np
import polars
import pytest
import xarray as xr

from ert import LibresFacade
from ert.config import GenDataConfig, SummaryConfig
from ert.storage import open_storage


def _create_gen_data_config_ds_and_obs(
    num_gen_data, num_gen_obs, num_indices, num_report_steps
):
    gen_data_keys = [f"gen_data_{i}" for i in range(num_gen_data)]
    gen_data_config = GenDataConfig(keys=gen_data_keys)

    gen_data_per_key = [
        polars.DataFrame(
            {
                "response_key": [f"{gen_data_keys[i]}"] * num_report_steps,
                "index": polars.Series(
                    [(j % num_indices) for j in range(num_report_steps)],
                    dtype=polars.UInt16,
                ),
                "report_step": polars.Series(
                    list(range(num_report_steps)), dtype=polars.UInt16
                ),
                "values": polars.Series(
                    [random.random() * 10 for _ in range(num_report_steps)],
                    dtype=polars.Float32,
                ),
            }
        )
        for i in range(num_gen_data)
    ]

    gen_data_df = polars.concat(gen_data_per_key)

    gen_data_obs = polars.DataFrame(
        {
            "response_key": polars.Series(
                [gen_data_keys[(i % num_gen_data)] for i in range(num_gen_obs)],
                dtype=polars.String,
            ),
            "observation_key": polars.Series(
                [f"gen_obs_{i}" for i in range(num_gen_obs)], dtype=polars.String
            ),
            "index": polars.Series(
                [random.randint(0, num_indices) for _ in range(num_gen_obs)],
                dtype=polars.UInt16,
            ),
            "report_step": polars.Series(
                [random.randint(0, num_report_steps) for _ in range(num_gen_obs)],
                dtype=polars.UInt16,
            ),
            "observations": polars.Series(
                [random.uniform(-100, 100) for _ in range(num_gen_obs)],
                dtype=polars.Float32,
            ),
            "std": polars.Series(
                [random.uniform(0, 1) for _ in range(num_gen_obs)],
                dtype=polars.Float32,
            ),
        }
    )

    return gen_data_config, gen_data_df, gen_data_obs


def _create_summary_config_ds_and_obs(
    num_summary_names, num_summary_timesteps, num_summary_obs
):
    summary_config = SummaryConfig(
        name="summary",
        keys=[f"sum_key_{i}" for i in range(num_summary_names)],
        input_file="",
    )

    summary_df = polars.DataFrame(
        {
            "response_key": [
                f"sum_key_{i % num_summary_names}"
                for i in range(num_summary_names * num_summary_timesteps)
            ],
            "time": [
                datetime.date(2010, 1, 1)
                + datetime.timedelta(days=10 * (i // num_summary_names))
                for i in range(num_summary_names * num_summary_timesteps)
            ],
            "values": polars.Series(
                [i for i in range(num_summary_names * num_summary_timesteps)],
                dtype=polars.UInt16,
            ),
        }
    )

    summary_obs_df = polars.DataFrame(
        {
            "response_key": summary_df["name"][:num_summary_obs],
            "observation_key": [f"sum_obs_{i}" for i in range(num_summary_obs)],
            "time": summary_df["time"][:num_summary_obs],
            "observations": polars.Series(
                [random.uniform(-100, 100) for _ in range(num_summary_obs)],
                dtype=polars.Float32,
            ),
            "std": polars.Series(
                [random.uniform(0, 1) for _ in range(num_summary_obs)],
                dtype=polars.Float32,
            ),
        }
    )

    return summary_config, summary_df, summary_obs_df


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    ("num_reals, num_gen_data, num_gen_obs, num_indices, num_report_steps"),
    [
        (100, 1, 1, 1, 1),
        (100, 5, 3, 2, 10),
        (10, 50, 100, 10, 200),
    ],
)
def test_unify_gen_data_correctness(
    tmpdir, num_reals, num_gen_data, num_gen_obs, num_indices, num_report_steps
):
    gen_data_config, gen_data_df, _ = _create_gen_data_config_ds_and_obs(
        num_gen_data, num_gen_obs, num_indices, num_report_steps
    )

    with open_storage(tmpdir, "w") as s:
        exp = s.create_experiment(
            responses=[gen_data_config],
        )

        ens = exp.create_ensemble(ensemble_size=num_reals, name="zero")
        to_concat = []
        for i in range(num_reals):
            to_concat.append(
                gen_data_df.clone().insert_column(0, polars.lit(i).alias("realization"))
            )
            ens.save_response("gen_data", gen_data_df.clone(), i)

        ens.combine_responses("gen_data")

        combined = ens.load_responses("gen_data", realizations=tuple(range(num_reals)))

        by_group = []
        for group, ds in gen_data_df.items():
            by_group.append(ds.expand_dims(name=[group]))

        ds_by_name = xr.concat(by_group, dim="name")
        by_real = []
        for i in range(num_reals):
            by_real.append(ds_by_name.expand_dims(realization=[i]))

        assert (
            xr.concat(by_real, dim="realization")
            .sortby("name")
            .equals(combined.sortby("name"))
        )


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    ("num_reals, num_summary_names, num_summary_timesteps, num_summary_obs"),
    [
        (2, 2, 2, 1),
        (100, 10, 200, 1),
        (500, 2, 200, 13),
        (50, 23, 173, 29),
    ],
)
def test_unify_summary_correctness(
    tmpdir, num_reals, num_summary_names, num_summary_timesteps, num_summary_obs
):
    summary_config, summary_ds, _ = _create_summary_config_ds_and_obs(
        num_summary_names, num_summary_timesteps, num_summary_obs
    )

    with open_storage(tmpdir, "w") as s:
        exp = s.create_experiment(responses=[summary_config])

        ens = exp.create_ensemble(ensemble_size=num_reals, name="zero")
        for i in range(num_reals):
            ens.save_response("summary", summary_ds, i)

        ens.combine_responses("summary")

        combined = ens.load_responses("summary", realizations=tuple(range(num_reals)))

        manual_concat = []
        for i in range(num_reals):
            manual_concat.append(summary_ds.expand_dims(realization=[i]))

        assert combined.equals(
            xr.combine_nested(manual_concat, concat_dim="realization")
        )


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    (
        "num_reals, num_summary_names, num_summary_timesteps, "
        "num_summary_obs, realizations_to_rewrite"
    ),
    [
        (2, 2, 2, 1, [0]),
        (7, 10, 200, 1, [0, 2, 4, 6]),
        (50, 10, 200, 1, [0, 10, 12, 15, 19, 21, 26, 29, 31, 33, 34, 39, 49]),
    ],
)
def test_rewrite_summary_for_some_realizations(
    tmpdir,
    num_reals,
    num_summary_names,
    num_summary_timesteps,
    num_summary_obs,
    realizations_to_rewrite,
):
    summary_config, summary_ds, _ = _create_summary_config_ds_and_obs(
        num_summary_names, num_summary_timesteps, num_summary_obs
    )

    with open_storage(tmpdir, "w") as s:
        exp = s.create_experiment(responses=[summary_config])

        ens = exp.create_ensemble(ensemble_size=num_reals, name="zero")
        for i in range(num_reals):
            ens.save_response("summary", summary_ds, i)

        ens.combine_responses("summary")

        combined = ens.load_responses("summary", realizations=tuple(range(num_reals)))

        manual_concat = []
        for i in range(num_reals):
            manual_concat.append(summary_ds.expand_dims(realization=[i]))

        assert combined.equals(
            xr.combine_nested(manual_concat, concat_dim="realization")
        )

        # Select out ds for reals to remove
        # Multiply values by 3
        scalefn = lambda x: x * 3
        selection_to_rewrite = combined.sel(realization=realizations_to_rewrite)
        scaled_selection_to_rewrite = selection_to_rewrite.map(scalefn)

        for real in realizations_to_rewrite:
            scaled_ds = combined.sel(realization=real).map(scalefn)
            ens.save_response("summary", scaled_ds, real)

            # Expect load_responses to give the value from the combined first
            assert ens.load_responses("summary", (real,)).equals(
                combined.sel(realization=[real])
            )

        ens.combine_responses("summary")

        # Now we expect them to be scaled
        for real in realizations_to_rewrite:
            # Now we expect scaled values
            assert ens.load_responses("summary", (real,)).equals(
                scaled_selection_to_rewrite.sel(realization=[real])
            )


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    (
        "num_reals, num_gen_data, num_gen_obs, num_indices, "
        "num_report_steps, realizations_to_rewrite"
    ),
    [
        (2, 1, 0, 1, 1, [0]),
        (5, 5, 0, 2, 10, [0, 2, 3, 4]),
        (50, 50, 0, 10, 200, [0, 2, 3, 5, 8, 9, 15, 23, 29, 39, 44]),
    ],
)
def test_rewrite_gen_data_for_some_realizations(
    tmpdir,
    num_reals,
    num_gen_data,
    num_gen_obs,
    num_indices,
    num_report_steps,
    realizations_to_rewrite,
):
    gen_data_configs, gen_data_ds, _ = _create_gen_data_config_ds_and_obs(
        num_gen_data, num_gen_obs, num_indices, num_report_steps
    )

    with open_storage(tmpdir, "w") as s:
        exp = s.create_experiment(
            responses=[*gen_data_configs],
        )

        ens = exp.create_ensemble(ensemble_size=num_reals, name="zero")
        for group, ds in gen_data_ds.items():
            for i in range(num_reals):
                ens.save_response(group, ds, i)

        ens.combine_responses("gen_data")

        combined = ens.load_responses("gen_data", realizations=tuple(range(num_reals)))

        by_group = []
        for group, ds in gen_data_ds.items():
            by_group.append(ds.expand_dims(name=[group]))

        ds_by_name = xr.concat(by_group, dim="name")
        by_real = []
        for i in range(num_reals):
            by_real.append(ds_by_name.expand_dims(realization=[i]))

        assert (
            xr.concat(by_real, dim="realization")
            .sortby("name")
            .equals(combined.sortby("name"))
        )

        # Select out ds for reals to remove
        # Multiply values by 3
        scalefn = lambda x: x * 3
        selection_to_rewrite = combined.sel(realization=realizations_to_rewrite).sel(
            name="gen_data_0", drop=True
        )
        scaled_selection_to_rewrite = selection_to_rewrite.map(scalefn)

        for real in realizations_to_rewrite:
            scaled_ds = selection_to_rewrite.sel(realization=real).map(scalefn)
            ens.save_response("gen_data_0", scaled_ds, real)

            # Expect load_responses to give the value from the combined first
            assert ens.load_responses("gen_data_0", (real,)).equals(
                selection_to_rewrite.sel(realization=[real])
            )

        ens.combine_responses("gen_data_0")

        # Now we expect them to be scaled
        for real in realizations_to_rewrite:
            # Now we expect scaled values
            assert ens.load_responses("gen_data_0", (real,)).equals(
                scaled_selection_to_rewrite.sel(realization=[real])
            )


@pytest.mark.usefixtures("copy_snake_oil_case_storage")
def test_that_unify_works_through_load_forward_model_with_realization_data_rewrites():
    facade = LibresFacade.from_config_file("snake_oil.ert")
    storage = open_storage(facade.enspath, mode="w")
    ensemble = storage.get_ensemble_by_name("default_0")

    facade.load_from_forward_model(
        ensemble, [True] * facade.config.model_config.num_realizations, 0
    )

    a_summary_key = "BPR:1,3,8"
    a_gen_data_key = "SNAKE_OIL_OPR_DIFF"

    reals_to_edit = [0, 2, 4]

    gen_datas_before_edit = [
        (real, ensemble.load_responses(a_gen_data_key, (real,)))
        for real in reals_to_edit
    ]
    summaries_before_edit = [
        (real, ensemble.load_responses("summary", (real,))) for real in reals_to_edit
    ]

    for real, ds in gen_datas_before_edit:
        edited_ds = ds.map(lambda _: 1337)
        ensemble.save_response(a_gen_data_key, edited_ds, real)

    for real, ds in summaries_before_edit:
        edited_ds = ds.map(lambda _: 1337)
        ensemble.save_response("summary", edited_ds, real)

    ensemble.combine_responses()

    def assert_unedited(realizations: List[int]):
        gen_data_combined_edited = ensemble.load_responses(
            a_gen_data_key, realizations=tuple(realizations)
        )

        assert set(
            gen_data_combined_edited.to_dataframe().values.reshape((-1)).tolist()
        ) != {1337}

        summary_combined_edited = ensemble.load_responses(
            a_summary_key, realizations=tuple(realizations)
        )

        assert set(
            summary_combined_edited.to_dataframe().values.reshape((-1)).tolist()
        ) != {1337}

    def assert_edited(realizations: List[int]):
        gen_data_combined_edited = ensemble.load_responses(
            a_gen_data_key, realizations=tuple(realizations)
        )

        assert set(
            gen_data_combined_edited.to_dataframe().values.reshape((-1)).tolist()
        ) == {1337}

        summary_combined_edited = ensemble.load_responses(
            a_summary_key, realizations=tuple(realizations)
        )

        assert set(
            summary_combined_edited.to_dataframe().values.reshape((-1)).tolist()
        ) == {1337}

    assert_edited(reals_to_edit)

    facade.load_from_forward_model(
        ensemble, np.ones(facade.config.model_config.num_realizations, dtype=bool), 0
    )

    # We still expect them to be edited as they aren't unified yet
    assert_edited(reals_to_edit)

    ensemble.combine_responses()

    # Now we expect unedited values in the ds again
    assert_unedited(reals_to_edit)
