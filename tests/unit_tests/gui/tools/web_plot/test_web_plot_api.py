import json
import math
import os
import pathlib
from datetime import date
from typing import Dict, List, TypedDict

import hypothesis.strategies as st
import numpy as np
import pandas as pd
import xarray
from hypothesis import given

from ert.gui.tools.web_plot.web_plot_server import (
    WebPlotServerConfig,
    WebPlotStorageAccessors,
    Experiment,
)

shared_uuids = st.shared(st.lists(st.uuids(), min_size=1, max_size=10, unique=True))
required_files = ["index.json", "responses.json"]


some_diff_kws = ["WPR", "OPR", "GPR", "WPR"]
some_param_kws = ["OP1", "OP2", "BPR", "WPR"]


def make_valid_parameters_file(path: pathlib.Path, json_contents: object):
    with open(path / "parameter.json", "w", encoding="utf-8") as f:
        json.dump(json_contents, f)


def make_valid_responses_json_file(path: str):
    diff_entries = {
        k: {
            "name": f"SUB_{k}_DIFF",
            "input_file": f"sub_{k.lower()}_diff_%d.txt",
            "report_steps": [199],
            "_ert_kind": "GenDataConfig",
        }
        for k in some_diff_kws
    }

    obj = {
        **diff_entries,
        "summary": {
            "name": "summary",
            "keys": ["FOPR", "FOPT"],
            "refcase": ["2015-01-04 00:00:00", "2012-09-16 00:00:00"],
            "input_file": "MINE_FIELD",
            "_ert_kind": "SummaryConfig",
        },
    }

    with open(os.path.join(path, "responses.json"), "w", encoding="utf-8") as f:
        json.dump(obj, f)


def make_valid_index_json_file(path, experiment_id: str, experiment_name: str):
    with open(os.path.join(path, "index.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "id": experiment_id,
                "name": experiment_name,
            },
            f,
        )


@st.composite
def transfer_function_definition_string_generator(draw):
    num_tfs = draw(st.integers(min_value=1, max_value=5))
    tf_names = draw(
        st.lists(
            st.sampled_from(
                [
                    "ATLE_JOHNNY",
                    "ATLE_BIRGER",
                    "TOM_BIRGER",
                    "JONNY_TOM",
                    "ARNLEIF",
                ]
            ),
            unique=True,
            min_size=num_tfs,
            max_size=num_tfs,
        ),
    )
    tf_distributions = draw(
        st.lists(
            st.sampled_from(
                [
                    "UNIFORM",
                    "NORMAL",
                    "LOGUNIF",
                    "DUNIF",
                    "CONST",
                    "DERFF",
                    "TRIANGULAR",
                ]
            ),
            min_size=num_tfs,
            max_size=num_tfs,
        )
    )
    tf_args = draw(
        st.lists(
            st.lists(st.floats(min_value=-1, max_value=1), min_size=2, max_size=5),
            min_size=num_tfs,
            max_size=num_tfs,
        )
    )

    return [
        f"{name.ljust(21)}{distribution} {' '.join([str(x) for x in args])}"
        for (name, distribution, args) in zip(tf_names, tf_distributions, tf_args)
    ]


@st.composite
def parameters_file_json_generator(draw):
    num_entries = draw(st.integers(min_value=1, max_value=3))
    names_list = draw(
        st.lists(
            st.sampled_from(["SNEK_OIL", "SNAK_OJIL", "MINE_FIELD", "WIND_FARM"]),
            unique=True,
            min_size=num_entries,
            max_size=num_entries,
        )
    )
    fwd_init_list = draw(
        st.lists(
            st.sampled_from(["true", "false"]),
            min_size=num_entries,
            max_size=num_entries,
        )
    )
    tf_defs_list = draw(
        st.lists(
            transfer_function_definition_string_generator(),
            min_size=num_entries,
            max_size=num_entries,
        )
    )

    return {
        name: {
            "name": name,
            "forward_init": fwd_init,
            "template_file": f"/some/where/templates/{name.lower()}_template.txt",
            "output_file": f"{name}_params.txt",
            "transfer_function_definitions": tf_defs,
            "forward_init_file": "null",
            "_ert_kind": "GenKwConfig",
        }
        for (name, fwd_init, tf_defs) in zip(names_list, fwd_init_list, tf_defs_list)
    }


@st.composite
def experiment_ensemble_setup(draw):
    exp_uuids = draw(shared_uuids)
    uuids_subset = draw(st.lists(st.sampled_from(exp_uuids), unique=True))
    files_to_remove = draw(
        st.lists(
            st.sampled_from([*required_files]),
            max_size=len(uuids_subset),
            min_size=len(uuids_subset),
        )
    )
    ensembles = draw(
        st.sets(st.uuids(), min_size=2, max_size=10).filter(
            lambda l: not any(x in exp_uuids for x in l)
        )
    )

    ensemble_experiment_links = draw(
        st.lists(
            st.sampled_from(exp_uuids), max_size=len(ensembles), min_size=len(ensembles)
        )
    )

    parameter_json = draw(
        st.lists(
            parameters_file_json_generator(),
            min_size=len(exp_uuids),
            max_size=len(exp_uuids),
        )
    )

    assert len(parameter_json) == len(exp_uuids)

    return (
        [str(s) for s in exp_uuids],
        list(zip([str(s) for s in uuids_subset], files_to_remove)),
        list(
            zip(
                [str(s) for s in ensembles], [str(s) for s in ensemble_experiment_links]
            )
        ),
        parameter_json,
    )


def create_experiment_storage(experiment_setup):
    all_experiments, invalid, ensembles, parameter_json_list = experiment_setup

    cwd = pathlib.Path(os.getcwd())
    os.mkdir(cwd / "experiments")
    experiments_folder = cwd / "experiments"
    for exp_uuid, parameter_json in zip(all_experiments, parameter_json_list):
        os.mkdir(experiments_folder / exp_uuid)
        make_valid_responses_json_file(experiments_folder / exp_uuid)
        make_valid_index_json_file(
            experiments_folder / exp_uuid,
            exp_uuid,
            f"exp_with_id_{exp_uuid}",
        )
        make_valid_parameters_file(experiments_folder / exp_uuid, parameter_json)


class RealizationInfo(TypedDict):
    exp_id: str
    ens_id: str
    index: int
    path: pathlib.Path


def create_ensemble_storage(
    all_experiments, ensembles, num_realizations: int = 25
) -> List[RealizationInfo]:
    cwd = pathlib.Path(os.getcwd())
    ensembles_folder = cwd / "ensembles"
    os.mkdir(ensembles_folder)
    prior_ensemble_ids = {x: "null" for x in all_experiments}

    realization_paths = []
    for i, (ens_id, exp_id) in enumerate(ensembles):
        ens_name = f"ensemble_{i}"
        prior_ensemble_id = prior_ensemble_ids[exp_id]
        prior_ensemble_ids[exp_id] = ens_id
        os.makedirs(ensembles_folder / ens_id)

        with open(
            ensembles_folder / ens_id / "index.json",
            "w+",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "id": ens_id,
                    "name": ens_name,
                    "experiment_id": exp_id,
                    "ensemble_size": num_realizations,
                    "iteration": i,
                    "prior_ensemble_id": prior_ensemble_id,
                    "started_at": str(date.today()),
                },
                f,
            )

        for i_real in range(num_realizations):
            os.mkdir(ensembles_folder / ens_id / f"realization-{i_real}")
            real_info: RealizationInfo = {
                "exp_id": exp_id,
                "ens_id": ens_id,
                "index": i_real,
                "path": ensembles_folder / ens_id / f"realization-{i_real}",
            }
            realization_paths.append(real_info)

    return realization_paths


def create_xarray_summary(num_keywords, num_timesteps):
    data = {
        'time': (
            pd.date_range(
                pd.to_datetime("2022-01-01"),
                pd.to_datetime("2022-01-01")
                + pd.to_timedelta(num_timesteps - 1, unit="D"),
            ).to_list()
            * num_keywords
        ),
        'name': [f"KW_{i}" for i in range(num_keywords) for _ in range(num_timesteps)],
        'values': np.random.uniform(0, 10, size=num_keywords * num_timesteps),
    }

    df = pd.DataFrame(data)

    # Convert the DataFrame to an xarray Dataset
    dataset = df.set_index(["name", "time"]).to_xarray()
    return dataset


@given(experiment_ensemble_setup())
def test_that_invalid_experiments_are_skipped(tmp_path_factory, experiment_setup):
    os.chdir(tmp_path_factory.mktemp("tmp"))
    all_experiments, invalid, ensembles, _ = experiment_setup
    cwd = pathlib.Path(os.getcwd())
    experiments_folder = cwd / "experiments"

    create_experiment_storage(experiment_setup)

    for exp_id, file_to_remove in invalid:
        os.remove(experiments_folder / exp_id / file_to_remove)

    create_ensemble_storage(all_experiments, ensembles)
    all_experiments, invalid, _, _ = experiment_setup

    config = WebPlotServerConfig(
        **{
            "directory_with_ert_storage": os.getcwd(),
            "directory_with_html": os.getcwd(),  # not used in this context
            "hostname": "localhost",
            "port": 9999,
        }
    )

    accessors = WebPlotStorageAccessors(config)

    meta = accessors.get_experiments_metadata()
    invalid_exp_ids = [exp_id for exp_id, _ in invalid]
    for v in all_experiments:
        if v not in invalid_exp_ids:
            assert v in meta

    for v in invalid_exp_ids:
        assert v[0] not in meta


@given(experiment_ensemble_setup())
def test_that_experiments_without_ensembles_are_skipped(
    tmp_path_factory, experiment_setup
):
    os.chdir(tmp_path_factory.mktemp("tmp"))
    all_experiments, invalid, ensembles, _ = experiment_setup

    create_experiment_storage(experiment_setup)

    # Do not create ensembles for experiments to be invalidated
    create_ensemble_storage(
        all_experiments,
        [
            (ens_id, exp_id)
            for (ens_id, exp_id) in ensembles
            if exp_id in [exp_id for exp_id, _ in invalid]
        ],
    )

    config = WebPlotServerConfig(
        **{
            "directory_with_ert_storage": os.getcwd(),
            "directory_with_html": os.getcwd(),  # not used in this context
            "hostname": "localhost",
            "port": 9999,
        }
    )

    accessors = WebPlotStorageAccessors(config)

    meta = accessors.get_experiments_metadata()

    invalid_exp_ids = [exp_id for exp_id, _ in invalid]
    for v in all_experiments:
        if v not in invalid_exp_ids:
            assert v in meta

    for v in invalid_exp_ids:
        assert v[0] not in meta


@given(
    experiment_ensemble_setup(),
    st.lists(st.booleans(), min_size=1, max_size=3),
    st.integers(min_value=1, max_value=5),
    st.integers(min_value=2, max_value=10),
)
def test_that_summary_load_works_with_missing_realizations(
    tmp_path_factory, experiment_setup, valid_realizations, num_keywords, num_timesteps
):
    os.chdir(tmp_path_factory.mktemp("tmp"))
    all_experiments, invalid, ensembles, _ = experiment_setup
    cwd = pathlib.Path(os.getcwd())

    create_experiment_storage(experiment_setup)
    create_ensemble_storage(
        all_experiments, ensembles, num_realizations=len(valid_realizations)
    )
    ensembles_folder = cwd / "ensembles"

    ensembles_by_experiment: Dict[str, List[str]] = {}
    for iens, (ens_id, exp_id) in enumerate(ensembles):
        if exp_id not in ensembles_by_experiment:
            ensembles_by_experiment[exp_id] = [ens_id]
        else:
            ensembles_by_experiment[exp_id].append(ens_id)

        for unshifted_i_real, is_valid in enumerate(valid_realizations):
            i_real = (unshifted_i_real + iens) % len(valid_realizations)
            # Shift exclude indices per-ensemble to achieve
            # a deterministic but non-uniform distribution of "failed"
            # realizations across different ensembles
            if not is_valid:  # Delete the folder
                os.rmdir(ensembles_folder / ens_id / f"realization-{i_real}")
            else:
                dataset = create_xarray_summary(num_keywords, num_timesteps)
                dataset.to_netcdf(
                    ensembles_folder / ens_id / f"realization-{i_real}" / "summary.nc"
                )

    config = WebPlotServerConfig(
        **{
            "directory_with_ert_storage": cwd,
            "directory_with_html": cwd,  # not used in this context
            "hostname": "localhost",
            "port": 9999,
        }
    )

    accessors = WebPlotStorageAccessors(config)

    for exp_id, ensembles in ensembles_by_experiment.items():
        for i in range(num_keywords):
            kw = f"KW_{i}"
            summary = accessors.get_summary_chart_data(ensembles, exp_id, kw)

            assert len(summary["data"]) == valid_realizations.count(True) * len(
                ensembles
            )

            for line in summary["data"]:
                assert len(line["points"]) == num_timesteps


@given(
    experiment_ensemble_setup(),
    st.lists(st.booleans(), min_size=1, max_size=3),
)
def test_that_parameters_meta_is_parsed(
    tmp_path_factory, experiment_setup, valid_realizations
):
    os.chdir(tmp_path_factory.mktemp("tmp"))
    all_experiments, _, all_ensembles, all_parameter_json = experiment_setup
    cwd = pathlib.Path(os.getcwd())

    create_experiment_storage(experiment_setup)
    create_ensemble_storage(
        all_experiments, all_ensembles, num_realizations=len(valid_realizations)
    )

    config = WebPlotServerConfig(
        **{
            "directory_with_ert_storage": cwd,
            "directory_with_html": cwd,  # not used in this context
            "hostname": "localhost",
            "port": 9999,
        }
    )

    accessors = WebPlotStorageAccessors(config)

    meta = accessors.get_experiments_metadata()
    for exp_id, parameter_json in zip(all_experiments, all_parameter_json):
        experiment = meta[exp_id]
        parameters = experiment.parameters
        assert parameter_json.keys() == parameters.keys()

        for key, spec in parameter_json.items():
            parsed = parameters[key]
            for i, tf_str in enumerate(spec["transfer_function_definitions"]):
                assert parsed.transfer_function_definitions[i].name in tf_str
                assert parsed.transfer_function_definitions[i].distribution in tf_str


def create_xarray_params_for_realization(tf_names, realization_index) -> xarray.Dataset:
    data = {
        "names": tf_names,
        "realizations": [realization_index] * len(tf_names),
        "values": np.random.uniform(-1, 1, size=len(tf_names)),
        "transformed_values": np.random.uniform(-4, 4, size=len(tf_names)),
    }

    df = pd.DataFrame(data)

    # Convert the DataFrame to an xarray Dataset
    dataset = df.set_index(["names", "realizations"]).to_xarray()
    return dataset


@given(
    experiment_ensemble_setup(),
    st.lists(st.booleans(), min_size=1, max_size=3),
)
def test_get_param_data(tmp_path_factory, experiment_setup, valid_realizations):
    os.chdir(tmp_path_factory.mktemp("tmp"))
    all_experiments, _, all_ensembles, all_parameter_json = experiment_setup
    cwd = pathlib.Path(os.getcwd())

    create_experiment_storage(experiment_setup)
    real_infos = create_ensemble_storage(
        all_experiments, all_ensembles, num_realizations=len(valid_realizations)
    )

    # Write the netcdf files for realizations
    exp_to_paramjson = {
        exp_id: param_json
        for exp_id, param_json in zip(all_experiments, all_parameter_json)
    }

    for real_info in real_infos:
        params_for_experiment = exp_to_paramjson[real_info["exp_id"]]
        for name, param_json in params_for_experiment.items():
            name = param_json["name"]
            path = real_info["path"] / (name + ".nc")
            tf_names = [
                x.split(" ")[0] for x in param_json["transfer_function_definitions"]
            ]
            ds = create_xarray_params_for_realization(tf_names, real_info["index"])
            ds.to_netcdf(path)

    config = WebPlotServerConfig(
        **{
            "directory_with_ert_storage": cwd,
            "directory_with_html": cwd,  # not used in this context
            "hostname": "localhost",
            "port": 9999,
        }
    )

    accessors = WebPlotStorageAccessors(config)

    meta = accessors.get_experiments_metadata()
    for exp_id, parameter_json in zip(all_experiments, all_parameter_json):
        experiment = meta[exp_id]
        total_num_reals = sum(
            [len(x.realizations) for x in experiment.ensembles.values()]
        )

        for param_name in experiment.all_parameter_keys:
            chart_data = accessors.get_parameter_chart_data(
                ensembles=list(experiment.ensembles.keys()),
                experiment_id=exp_id,
                parameter=param_name,
            )
            assert len(chart_data["data"]) == total_num_reals
            for d in chart_data["data"]:
                assert not math.isnan(d["values"])
                assert not math.isnan(d["transformed_values"])
                assert isinstance(d["values"], float)
                assert isinstance(d["transformed_values"], float)
