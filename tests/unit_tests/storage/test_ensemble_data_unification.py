import datetime
import random
from typing import List

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ert import LibresFacade
from ert.config import GenDataConfig, GenKwConfig, SummaryConfig
from ert.config.gen_kw_config import TransformFunctionDefinition
from ert.storage import open_storage


def _create_param_config_and_ds(num_groups, num_tfs):
    configs = [
        GenKwConfig(
            name=f"param_{i}",
            transform_function_definitions=[
                TransformFunctionDefinition(
                    name=f"tf_{j}", param_name="UNIFORM", values=[0, 1]
                )
                for j in range(num_tfs)
            ],
            template_file="",
            output_file="",
            forward_init_file=None,
            update=True,
            forward_init=False,
        )
        for i in range(num_groups)
    ]

    ds_name_tuples = [
        (
            f"param_{i}",
            xr.Dataset(
                {
                    "values": ("names", [i, i, i]),
                    "transformed_values": ("names", [i, i, i]),
                    "names": ["a", "b", "c"],
                }
            ),
        )
        for i in range(num_groups)
    ]

    return configs, ds_name_tuples


def _create_gen_data_config_ds_and_obs(
    num_gen_data, num_gen_obs, num_indices, num_report_steps
):
    gen_data_configs = [
        *[GenDataConfig(name=f"gen_data_{i}") for i in range(num_gen_data)]
    ]

    gen_data_ds = {
        f"{gen_data_configs[i].name}": pd.DataFrame(
            data={
                "index": [(j % num_indices) for j in range(num_report_steps)],
                "report_step": list(range(num_report_steps)),
                "values": [random.random() * 10 for _ in range(num_report_steps)],
            }
        )
        .set_index(["index", "report_step"])
        .to_xarray()
        for i in range(num_gen_data)
    }

    gen_data_obs = (
        pd.DataFrame(
            data={
                "name": [
                    gen_data_configs[(i % num_gen_data)].name
                    for i in range(num_gen_obs)
                ],
                "obs_name": [f"gen_obs_{i}" for i in range(num_gen_obs)],
                "index": [
                    f"{random.randint(0,num_indices)}" for _ in range(num_gen_obs)
                ],
                "report_step": [
                    random.randint(0, num_report_steps) for _ in range(num_gen_obs)
                ],
                "observations": [random.uniform(-100, 100) for _ in range(num_gen_obs)],
                "std": [random.uniform(0, 1) for _ in range(num_gen_obs)],
            }
        )
        .set_index(["name", "obs_name", "index", "report_step"])
        .to_xarray()
    )

    gen_data_obs.attrs["response"] = "gen_data"

    return gen_data_configs, gen_data_ds, gen_data_obs


def _create_summary_config_ds_and_obs(
    num_summary_names, num_summary_timesteps, num_summary_obs
):
    summary_config = SummaryConfig(
        name="summary",
        keys=[f"sum_key_{i}" for i in range(num_summary_names)],
        input_file="",
    )

    summary_df = pd.DataFrame(
        data={
            "time": [
                pd.to_datetime(
                    datetime.date(2010, 1, 1)
                    + datetime.timedelta(days=10 * (i // num_summary_names))
                )
                for i in range(num_summary_names * num_summary_timesteps)
            ],
            "name": [
                f"sum_key_{i%num_summary_names}"
                for i in range(num_summary_names * num_summary_timesteps)
            ],
            "values": list(range(num_summary_names * num_summary_timesteps)),
        }
    )

    summary_obs_ds = (
        pd.DataFrame(
            data={
                "time": [summary_df.loc[i]["time"] for i in range(num_summary_obs)],
                "name": [summary_df.loc[i]["name"] for i in range(num_summary_obs)],
                "obs_name": [f"sum_obs_{i}" for i in range(num_summary_obs)],
                "observations": [
                    random.uniform(-100, 100) for _ in range(num_summary_obs)
                ],
                "std": [random.uniform(0, 1) for _ in range(num_summary_obs)],
            }
        )
        .set_index(["name", "obs_name", "time"])
        .to_xarray()
    )
    summary_obs_ds.attrs["response"] = "summary"

    summary_ds = summary_df.set_index(["name", "time"]).to_xarray()

    return summary_config, summary_ds, summary_obs_ds


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    ("num_reals, num_responses, num_obs, num_indices, num_report_steps"),
    [
        (100, 1, 1, 1, 1),
        (100, 5, 3, 2, 10),
        (10, 50, 100, 10, 200),
    ],
)
def test_that_observation_getters_from_experiment_match_expected_data(
    tmpdir, num_reals, num_responses, num_obs, num_indices, num_report_steps
):
    gen_data_configs, gen_data_ds, gen_data_obs = _create_gen_data_config_ds_and_obs(
        num_responses, num_obs, num_indices, num_report_steps
    )

    summary_config, summary_ds, summary_obs = _create_summary_config_ds_and_obs(
        num_responses, num_indices * num_report_steps, num_obs
    )

    with open_storage(tmpdir, "w") as s:
        exp = s.create_experiment(
            responses=[*gen_data_configs],
            observations={"gen_data": gen_data_obs, "summary": summary_obs},
        )

        assert exp._obs_key_to_response_name_and_type == {
            **{
                f"gen_obs_{i}": (
                    f"gen_data_{i%num_responses}",
                    "gen_data",
                )
                for i in range(num_obs)
            },
            **{
                f"sum_obs_{i}": (f"sum_key_{i%num_responses}", "summary")
                for i in range(num_obs)
            },
        }

        for i in range(num_obs):
            assert (
                gen_data_obs.sel(obs_name=f"gen_obs_{i}", drop=True)
                .dropna("name", how="all")
                .squeeze("name", drop=True)
                .to_dataframe()
                .dropna()
                .equals(exp.get_single_obs_ds(f"gen_obs_{i}").to_dataframe().dropna())
            )
            assert (
                summary_obs.sel(obs_name=f"sum_obs_{i}", drop=True)
                .dropna("name", how="all")
                .squeeze("name", drop=True)
                .to_dataframe()
                .dropna()
                .equals(exp.get_single_obs_ds(f"sum_obs_{i}").to_dataframe().dropna())
            )


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

        ens.unify_responses("gen_data")

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

        ens.unify_responses("summary")

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

        ens.unify_responses("summary")

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

        ens.unify_responses("summary")

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

        ens.unify_responses("gen_data")

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

        ens.unify_responses("gen_data_0")

        # Now we expect them to be scaled
        for real in realizations_to_rewrite:
            # Now we expect scaled values
            assert ens.load_responses("gen_data_0", (real,)).equals(
                scaled_selection_to_rewrite.sel(realization=[real])
            )


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    ("num_reals, num_groups, num_tfs, realizations_to_rewrite"),
    [
        (2, 1, 3, [0]),
        (5, 5, 3, [0, 2, 3, 4]),
        (50, 50, 3, [0, 2, 3, 5, 8, 9, 15, 23, 29, 39, 44]),
    ],
)
def test_unify_parameters_correctness(
    tmpdir,
    num_reals,
    num_groups,
    num_tfs,
    realizations_to_rewrite,
):
    configs, name_ds_tuples = _create_param_config_and_ds(num_groups, num_tfs)

    with open_storage(tmpdir, "w") as s:
        exp = s.create_experiment(parameters=configs)

        ens = exp.create_ensemble(ensemble_size=num_reals, name="zero")

        combined_datasets = {}
        for group, ds in name_ds_tuples:
            combined_datasets[group] = []
            for real in range(num_reals):
                ens.save_parameters(group, real, ds)
                combined_datasets[group].append(ds.expand_dims(realizations=[real]))

            combined_datasets[group] = xr.concat(
                combined_datasets[group], dim="realizations"
            )

        for grp, ds in combined_datasets.items():
            loaded_ds = ens.load_parameters(grp)
            ds = ds.reindex_like(loaded_ds)
            assert loaded_ds.equals(ds)

        ens.unify_parameters()
        for grp, ds in combined_datasets.items():
            loaded_ds = ens.load_parameters(grp)
            ds = ds.reindex_like(loaded_ds)
            assert loaded_ds.equals(ds)

        # Now remove some parameters
        added_datasets = []
        for group, ds in name_ds_tuples:
            for real in range(num_reals):
                if real in realizations_to_rewrite:
                    scaled_ds = ds.map(lambda x: x * 3)
                    ens.save_parameters(group, real, scaled_ds)
                    added_datasets.append((group, real, scaled_ds))
                else:
                    added_datasets.append((group, real, ds))

        ens.unify_parameters()

        for group, realization, ds in added_datasets:
            loaded_ds = ens.load_parameters(group, realization)
            assert loaded_ds.equals(ds.reindex_like(loaded_ds))


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

    ensemble.unify_responses()

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

    ensemble.unify_responses()

    # Now we expect unedited values in the ds again
    assert_unedited(reals_to_edit)
