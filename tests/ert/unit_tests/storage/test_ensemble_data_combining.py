import datetime
import random
from typing import List

import numpy as np
import polars
import pytest

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
        input_files=[""],
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
                [*range(num_summary_names * num_summary_timesteps)],
                dtype=polars.Float32,
            ),
        }
    )

    summary_obs_df = polars.DataFrame(
        {
            "response_key": summary_df["response_key"][:num_summary_obs],
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
        expected_single_response_dfs = []
        for i in range(num_reals):
            expected_single_response_df = gen_data_df.with_columns(
                0, polars.lit(i, dtype=polars.UInt16).alias("realization")
            ).select(["realization", *gen_data_df.columns])
            expected_single_response_dfs.append(expected_single_response_df)
            ens.save_response("gen_data", gen_data_df.clone(), i)
            assert ens.load_responses("gen_data", (i,)).equals(
                expected_single_response_df
            )

        ens.combine_responses("gen_data")
        ens.load_responses.cache_clear()

        for i in range(num_reals):
            assert ens.load_responses("gen_data", (i,)).equals(
                expected_single_response_dfs[i]
            )

        combined = ens.load_responses("gen_data", realizations=tuple(range(num_reals)))
        assert combined.equals(polars.concat(expected_single_response_dfs))


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
    summary_config, summary_df, _ = _create_summary_config_ds_and_obs(
        num_summary_names, num_summary_timesteps, num_summary_obs
    )

    with open_storage(tmpdir, "w") as s:
        exp = s.create_experiment(responses=[summary_config])

        ens = exp.create_ensemble(ensemble_size=num_reals, name="zero")

        expected_single_response_dfs = []
        for i in range(num_reals):
            expected_single_response_df = summary_df.with_columns(
                0, polars.lit(i, dtype=polars.UInt16).alias("realization")
            ).select(["realization", *summary_df.columns])
            expected_single_response_dfs.append(expected_single_response_df)
            ens.save_response("summary", summary_df.clone(), i)
            assert ens.load_responses("summary", (i,)).equals(
                expected_single_response_df
            )

        ens.combine_responses("summary")

        ens.load_responses.cache_clear()
        for i in range(num_reals):
            assert ens.load_responses("summary", (i,)).equals(
                expected_single_response_dfs[i]
            )

        combined = ens.load_responses("summary", realizations=tuple(range(num_reals)))
        assert combined.equals(polars.concat(expected_single_response_dfs))


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
    summary_config, summary_df, _ = _create_summary_config_ds_and_obs(
        num_summary_names, num_summary_timesteps, num_summary_obs
    )

    with open_storage(tmpdir, "w") as s:
        exp = s.create_experiment(responses=[summary_config])

        ens = exp.create_ensemble(ensemble_size=num_reals, name="zero")

        expected_single_response_dfs = []
        for i in range(num_reals):
            expected_single_response_df = summary_df.with_columns(
                0, polars.lit(i, dtype=polars.UInt16).alias("realization")
            ).select(["realization", *summary_df.columns])
            expected_single_response_dfs.append(expected_single_response_df)
            ens.save_response("summary", summary_df.clone(), i)
            assert ens.load_responses("summary", (i,)).equals(
                expected_single_response_df
            )

        ens.combine_responses("summary")

        ens.load_responses.cache_clear()
        for i in range(num_reals):
            assert ens.load_responses("summary", (i,)).equals(
                expected_single_response_dfs[i]
            )

        combined = ens.load_responses("summary", realizations=tuple(range(num_reals)))
        assert combined.equals(polars.concat(expected_single_response_dfs))

        ens.load_responses.cache_clear()
        # Select out ds for reals to remove
        # Multiply values by 3
        selection_to_rewrite = combined.filter(
            polars.col("realization").is_in(tuple(realizations_to_rewrite))
        )
        scaled_selection_to_rewrite = selection_to_rewrite.with_columns(
            (polars.col("values") * 3).alias("values")
        )

        for real in realizations_to_rewrite:
            scaled_ds = scaled_selection_to_rewrite.filter(
                polars.col("realization").eq(real)
            )
            ens.save_response("summary", scaled_ds, real)

            # Expect load_responses to give the value from the combined first
            assert ens.load_responses("summary", (real,)).equals(
                combined.filter(polars.col("realization").eq(real))
            )

        ens.combine_responses("summary")

        ens.load_responses.cache_clear()

        # Now we expect them to be scaled
        for real in realizations_to_rewrite:
            # Now we expect scaled values
            assert ens.load_responses("summary", (real,)).equals(
                scaled_selection_to_rewrite.filter(polars.col("realization").eq(real))
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
    gen_data_config, gen_data_df, _ = _create_gen_data_config_ds_and_obs(
        num_gen_data, num_gen_obs, num_indices, num_report_steps
    )

    with open_storage(tmpdir, "w") as s:
        exp = s.create_experiment(
            responses=[gen_data_config],
        )

        ens = exp.create_ensemble(ensemble_size=num_reals, name="zero")
        expected_single_response_dfs = []
        for i in range(num_reals):
            expected_single_response_df = gen_data_df.with_columns(
                0, polars.lit(i, dtype=polars.UInt16).alias("realization")
            ).select(["realization", *gen_data_df.columns])
            expected_single_response_dfs.append(expected_single_response_df)
            ens.save_response("gen_data", gen_data_df.clone(), i)
            assert ens.load_responses("gen_data", (i,)).equals(
                expected_single_response_df
            )

        ens.combine_responses("gen_data")

        ens.load_responses.cache_clear()
        for i in range(num_reals):
            assert ens.load_responses("gen_data", (i,)).equals(
                expected_single_response_dfs[i]
            )

        combined = ens.load_responses("gen_data", realizations=tuple(range(num_reals)))
        assert combined.equals(polars.concat(expected_single_response_dfs))

        ens.load_responses.cache_clear()
        # Select out ds for reals to remove
        # Multiply values by 3
        selection_to_rewrite = combined.filter(
            polars.col("realization").is_in(tuple(realizations_to_rewrite))
        )
        scaled_selection_to_rewrite = selection_to_rewrite.with_columns(
            (polars.col("values") * 3).alias("values")
        )

        for real in realizations_to_rewrite:
            scaled_ds = scaled_selection_to_rewrite.filter(
                polars.col("realization").eq(real)
            )
            ens.save_response("gen_data", scaled_ds, real)

            # Expect load_responses to give the value from the combined first
            assert ens.load_responses("gen_data", (real,)).equals(
                combined.filter(polars.col("realization").eq(real))
            )

        ens.combine_responses("gen_data")

        ens.load_responses.cache_clear()

        # Now we expect them to be scaled
        for real in realizations_to_rewrite:
            # Now we expect scaled values
            assert ens.load_responses("gen_data", (real,)).equals(
                scaled_selection_to_rewrite.filter(polars.col("realization").eq(real))
            )


@pytest.mark.usefixtures("copy_snake_oil_case_storage")
def test_that_unify_works_through_load_forward_model_with_realization_data_rewrites():
    facade = LibresFacade.from_config_file("snake_oil.ert")
    storage = open_storage(facade.enspath, mode="w")
    experiment = next(storage.experiments)
    ensemble = experiment.get_ensemble_by_name("default_0")

    facade.load_from_forward_model(
        ensemble, [True] * facade.config.model_config.num_realizations, 0
    )

    a_summary_key = "BPR:1,3,8"
    a_gen_data_key = "SNAKE_OIL_OPR_DIFF"

    reals_to_edit = [0, 2, 4]

    gen_datas_before_edit = [
        (real, ensemble.load_responses("gen_data", (real,))) for real in reals_to_edit
    ]
    summaries_before_edit = [
        (real, ensemble.load_responses("summary", (real,))) for real in reals_to_edit
    ]

    for real, df in gen_datas_before_edit:
        edited_df = df.with_columns(
            polars.when(polars.col("response_key").eq(a_gen_data_key))
            .then(polars.lit(1337, dtype=polars.Float32))
            .otherwise(polars.col("values"))
            .alias("values")
        )

        ensemble.save_response(
            "gen_data",
            edited_df,
            real,
        )

    for real, df in summaries_before_edit:
        edited_df = df.with_columns(
            polars.when(polars.col("response_key").eq(a_summary_key))
            .then(polars.lit(1337, dtype=polars.Float32))
            .otherwise(polars.col("values"))
            .alias("values")
        )

        ensemble.save_response(
            "summary",
            edited_df,
            real,
        )

    ensemble.combine_responses()

    def assert_unedited(realizations: List[int]):
        gen_data_combined_edited = ensemble.load_responses(
            "gen_data", realizations=tuple(realizations)
        ).filter(polars.col("response_key").eq(a_gen_data_key))

        assert set(gen_data_combined_edited["values"].unique()) != {1337}

        summary_combined_edited = ensemble.load_responses(
            "summary", realizations=tuple(realizations)
        ).filter(polars.col("response_key").eq(a_summary_key))

        assert set(summary_combined_edited["values"].unique()) != {1337}

    def assert_edited(realizations: List[int]):
        gen_data_combined_edited = ensemble.load_responses(
            "gen_data", realizations=tuple(realizations)
        ).filter(polars.col("response_key").eq(a_gen_data_key))

        assert set(gen_data_combined_edited["values"].unique()) == {1337}

        summary_combined_edited = ensemble.load_responses(
            "summary", realizations=tuple(realizations)
        ).filter(polars.col("response_key").eq(a_summary_key))

        assert set(summary_combined_edited["values"].unique()) == {1337}

    assert_edited(reals_to_edit)

    facade.load_from_forward_model(
        ensemble, np.ones(facade.config.model_config.num_realizations, dtype=bool), 0
    )

    # We still expect them to be edited as they aren't unified yet
    assert_edited(reals_to_edit)

    ensemble.combine_responses()
    ensemble.load_responses.cache_clear()

    # Now we expect unedited values in the ds again
    assert_unedited(reals_to_edit)
