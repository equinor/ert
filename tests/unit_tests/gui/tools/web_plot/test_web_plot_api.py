import json
import os
import pathlib
from datetime import date
from typing import Dict, List

import hypothesis.strategies as st
import numpy as np
import pandas as pd
from hypothesis import given

from ert.gui.tools.web_plot.web_plot_server import (
    WebPlotServerConfig,
    WebPlotStorageAccessors,
)

shared_uuids = st.shared(st.lists(st.uuids(), min_size=1, max_size=10, unique=True))
required_files = ["index.json", "responses.json"]


some_diff_kws = ["WPR", "OPR", "GPR", "WPR"]
some_param_kws = ["OP1", "OP2", "BPR", "WPR"]


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
def experiment_ensemble_setup(draw):
    uuids = draw(shared_uuids)
    uuids_subset = draw(st.lists(st.sampled_from(uuids), unique=True))
    files_to_remove = draw(
        st.lists(
            st.sampled_from([*required_files]),
            max_size=len(uuids_subset),
            min_size=len(uuids_subset),
        )
    )
    ensembles = draw(
        st.sets(st.uuids(), min_size=2, max_size=10).filter(
            lambda l: not any(x in uuids for x in l)
        )
    )

    ensemble_experiment_links = draw(
        st.lists(
            st.sampled_from(uuids), max_size=len(ensembles), min_size=len(ensembles)
        )
    )

    return (
        [str(s) for s in uuids],
        list(zip([str(s) for s in uuids_subset], files_to_remove)),
        list(
            zip(
                [str(s) for s in ensembles], [str(s) for s in ensemble_experiment_links]
            )
        ),
    )


def create_experiment_storage(experiment_setup):
    all_experiments, invalid, ensembles = experiment_setup

    cwd = pathlib.Path(os.getcwd())
    os.mkdir(cwd / "experiments")
    experiments_folder = cwd / "experiments"
    for exp_uuid in all_experiments:
        os.mkdir(experiments_folder / exp_uuid)
        make_valid_responses_json_file(experiments_folder / exp_uuid)
        make_valid_index_json_file(
            experiments_folder / exp_uuid,
            exp_uuid,
            f"exp_with_id_{exp_uuid}",
        )


def create_ensemble_storage(all_experiments, ensembles, num_realizations: int = 25):
    cwd = pathlib.Path(os.getcwd())
    ensembles_folder = cwd / "ensembles"
    os.mkdir(ensembles_folder)
    prior_ensemble_ids = {x: "null" for x in all_experiments}
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
    tmp = tmp_path_factory.mktemp("tmp")
    os.chdir(tmp)
    all_experiments, invalid, ensembles = experiment_setup
    cwd = pathlib.Path(os.getcwd())
    experiments_folder = cwd / "experiments"

    create_experiment_storage(experiment_setup)

    for exp_id, file_to_remove in invalid:
        os.remove(experiments_folder / exp_id / file_to_remove)

    create_ensemble_storage(all_experiments, ensembles)
    all_experiments, invalid, _ = experiment_setup

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
    tmp = tmp_path_factory.mktemp("tmp")
    os.chdir(tmp)
    all_experiments, invalid, ensembles = experiment_setup

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
    tmp = tmp_path_factory.mktemp("tmp")
    os.chdir(tmp)
    all_experiments, invalid, ensembles = experiment_setup
    cwd = pathlib.Path(os.getcwd())

    create_experiment_storage(experiment_setup)
    create_ensemble_storage(
        all_experiments, ensembles, num_realizations=len(valid_realizations)
    )
    all_experiments, invalid, _ = experiment_setup
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
            summary = accessors.get_summary_chart_data(",".join(ensembles), exp_id, kw)

            assert len(summary["data"]) == valid_realizations.count(True) * len(
                ensembles
            )

            for line in summary["data"]:
                assert len(line["points"]) == num_timesteps
