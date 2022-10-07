import io
import json
import random
import string
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal

from ert import LibresFacade
from ert._c_wrappers.enkf import RunContext
from ert.shared.storage import extraction


@pytest.mark.parametrize(
    "x_axis, expected",
    [
        ([1, 2, 3, 4], ["1", "2", "3", "4"]),
        (["a", "b", "c"], ["a", "b", "c"]),
        (
            [pd.Timestamp(x, unit="d") for x in range(4)],
            [
                "1970-01-01T00:00:00",
                "1970-01-02T00:00:00",
                "1970-01-03T00:00:00",
                "1970-01-04T00:00:00",
            ],
        ),
    ],
)
def test_prepare_x_axis(x_axis, expected):
    assert expected == extraction._prepare_x_axis(x_axis)


class ErtConfigBuilder:
    def __init__(self):
        self.ensemble_size = 1
        self._priors = {}
        self._obs = []

    def add_general_observation(self, observation_name, response_name, data):
        """Add GENERAL_OBSERVATION

        The `data` parameter is a pandas DataFrame. This is to be a two-column
        frame where the first column are the values and the second column are
        the errors. The index-column of this frame are the observation's indices
        which link the observations to the responses.

        """
        self._obs.append((observation_name, response_name, data.copy()))

    def add_prior(self, name, entry):
        assert name not in self._priors
        self._priors[name] = entry
        return self

    def build(self, path=None):
        if path is None:
            path = Path.cwd()

        self._build_ert(path)
        self._build_job(path)
        self._build_observations(path)
        self._build_priors(path)

        return LibresFacade.from_config_file(str(path / "test.ert"))

    def _build_ert(self, path):
        f = (path / "test.ert").open("w")

        # Default
        f.write(
            "JOBNAME poly_%d\n"
            "QUEUE_SYSTEM LOCAL\n"
            "QUEUE_OPTION LOCAL MAX_RUNNING 50\n"
            f"NUM_REALIZATIONS {self.ensemble_size}\n"
        )

    def _build_job(self, path):
        (path / "test.ert").open("a")

    def _build_observations(self, path):
        """
        Creates a TIME_MAP and OBS_CONFIG entry in the ERT config. The TIME_MAP
        is required for ERT to load the OBS_CONFIG.

        Creates an 'obs_config.txt' file into which the generate observations
        are written.
        """
        if not self._obs:
            return

        (path / "time_map").write_text("2006-10-01\n")
        with (path / "test.ert").open("a") as f:
            f.write("OBS_CONFIG obs_config.txt\n")
            f.write("TIME_MAP time_map\n")
            f.write(
                (
                    "GEN_DATA RES RESULT_FILE:poly_%d.out "
                    "REPORT_STEPS:0 INPUT_FORMAT:ASCII\n"
                )
            )

        with (path / "obs_config.txt").open("w") as f:
            for obs_name, resp_name, data in self._obs:
                indices = ",".join(str(index) for index in data.index.tolist())

                f.write(
                    f"GENERAL_OBSERVATION {obs_name} {{\n"
                    f"    DATA       = {resp_name};\n"
                    f"    INDEX_LIST = {indices};\n"
                    f"    RESTART    = 0;\n"
                    f"    OBS_FILE   = {obs_name}.txt;\n"
                    "};\n"
                )
                with (path / f"{obs_name}.txt").open("w") as fobs:
                    data.to_csv(fobs, sep=" ", header=False, index=False)

    def _build_priors(self, path):
        if not self._priors:
            return

        with (path / "test.ert").open("a") as f:
            f.write("GEN_KW COEFFS coeffs.json.in coeffs.json coeffs_priors\n")
        with (path / "coeffs.json.in").open("w") as f:
            f.write("{\n")
            f.write(",\n".join(f'  "{name}": <{name}>' for name in self._priors))
            f.write("\n}\n")
        with (path / "coeffs_priors").open("w") as f:
            for name, entry in self._priors.items():
                f.write(f"{name} {entry}\n")


@pytest.fixture(autouse=True)
def _chdir_tmp_path(monkeypatch, tmp_path):
    """
    All tests in this file must be run in a clean directory
    """
    monkeypatch.chdir(tmp_path)


def test_empty_ensemble(client):
    ert = ErtConfigBuilder().build()
    extraction.post_ensemble_data(ert, -1)

    _id = client.fetch_experiment()

    # Name is "default"
    for ens in client.get(f"/experiments/{_id}/ensembles").json():
        assert (
            client.get(f"/ensembles/{ens['id']}/userdata").json()["name"] == "default"
        )

    # No priors exist
    assert client.get(f"/experiments/{_id}").json()["priors"] == {}


def test_empty_ensemble_with_name(client):
    name = _rand_name()

    # Create case with given name
    ert = ErtConfigBuilder().build()
    ert.select_or_create_new_case(name)

    # Post initial ensemble
    extraction.post_ensemble_data(ert, -1)

    # Compare results
    _id = client.fetch_experiment()
    for ens in client.get(f"/experiments/{_id}/ensembles").json():
        assert client.get(f"/ensembles/{ens['id']}/userdata").json()["name"] == name


def test_priors(client):
    priors = _make_priors()

    # Add priors to ERT config
    builder = ErtConfigBuilder()
    for name, conf, _ in priors:
        builder.add_prior(name, conf)
    ert = builder.build()

    # Start ERT
    _create_runpath(ert)

    # Post initial ensemble
    extraction.post_ensemble_data(ert, -1)

    # Compare results
    _id = client.fetch_experiment()
    actual_priors = client.get(f"/experiments/{_id}").json()["priors"]
    assert len(priors) == len(actual_priors)
    for name, _, resp in priors:
        assert actual_priors[f"COEFFS:{name}"] == resp


def test_parameters(client):
    priors = _make_priors()

    # Add priors to ERT config
    builder = ErtConfigBuilder()
    builder.ensemble_size = 10
    for name, conf, _ in priors:
        builder.add_prior(name, conf)
    ert = builder.build()

    # Start ERT
    _create_runpath(ert)

    # Post initial ensemble
    extraction.post_ensemble_data(ert, -1)

    # Get ensemble_id
    experiment_id = client.fetch_experiment()
    ensembles = client.get(f"/experiments/{experiment_id}/ensembles").json()
    ensemble_id = ensembles[0]["id"]

    # Compare parameters (+ 2 due to the two log10_ coeffs)
    parameters = client.get(f"/ensembles/{ensemble_id}/parameters").json()
    parameter_names = [entry["name"] for entry in parameters]
    assert len(parameters) == len(parameter_names)
    assert len(parameters) == len(priors) + 2
    for name, _, prior in priors:
        assert f"COEFFS:{name}" in parameter_names
        if prior["function"] in ("lognormal", "loguniform"):
            assert f"LOG10_COEFFS:{name}" in parameter_names

    # Compare records (+ 2 due to the two log10_ coeffs)
    records = client.get(f"/ensembles/{ensemble_id}/records").json()
    assert len(records) == len(priors) + 2
    for name, _, prior in priors:
        assert f"COEFFS:{name}" in records
        if prior["function"] in ("lognormal", "loguniform"):
            assert f"LOG10_COEFFS:{name}" in records

    parameters_df = _get_parameters()
    assert len(parameters_df) == builder.ensemble_size
    for col in parameters_df:
        record_data = client.get(
            f"/ensembles/{ensemble_id}/records/COEFFS:{col}",
            headers={"accept": "application/x-parquet"},
        ).content
        stream = io.BytesIO(record_data)
        df = pd.read_parquet(stream)

        # ERT produces a low-quality version
        assert_almost_equal(parameters_df[col].values, df.values.flatten(), decimal=4)


def test_observations(client):
    data = pd.DataFrame([[1, 0.1], [2, 0.2], [3, 0.3], [4, 0.4]], index=[2, 4, 6, 8])

    builder = ErtConfigBuilder()
    builder.add_general_observation("OBS", "RES", data)
    ert = builder.build()

    # Post ensemble
    extraction.post_ensemble_data(ert, builder.ensemble_size)

    # Experiment should have 1 observation
    experiment_id = client.fetch_experiment()
    observations = client.get(f"/experiments/{experiment_id}/observations").json()
    assert len(observations) == 1

    # Validate data
    obs = observations[0]
    assert obs["name"] == "OBS"
    assert obs["values"] == data[0].tolist()
    assert obs["errors"] == data[1].tolist()
    assert obs["x_axis"] == data.index.astype(str).tolist()
    assert obs["transformation"] is None


def test_observation_transformation(client):
    data = pd.DataFrame([[1, 0.1], [2, 0.2], [3, 0.3], [4, 0.4]], index=[0, 1, 2, 3])

    builder = ErtConfigBuilder()
    builder.ensemble_size = 5
    builder.add_general_observation("OBS", "RES", data)
    builder.add_prior("PARAMETER", "NORMAL 0.4 0.07")
    ert = builder.build()

    # Post first ensemble
    parent_ensemble_id = extraction.post_ensemble_data(ert, builder.ensemble_size)

    # Create runpath and run ERT
    run_context = _create_runpath(ert)
    for nr, path in enumerate(run_context.paths):
        (Path(path) / "poly_0.out").write_text(
            f"{1000 + nr}\n{1 + nr}\n{2 + nr}\n{3}\n"
        )
    ert.load_from_forward_model("default", [True] * len(run_context), 0)
    ert.smoother_update(run_context)

    # Post second ensemble
    update_id = extraction.post_update_data(ert, parent_ensemble_id, "boruvka")
    child_ensemble_id = extraction.post_ensemble_data(
        ert, builder.ensemble_size, update_id
    )

    # Ensemble should have 1 observation with transformation
    observations = client.get(f"/ensembles/{child_ensemble_id}/observations").json()
    assert len(observations) == 1

    # Validate data
    obs = observations[0]
    assert obs["name"] == "OBS"
    assert obs["values"] == data[0].tolist()
    assert obs["errors"] == data[1].tolist()

    trans = obs["transformation"]
    assert trans["name"] == "OBS"
    assert trans["active"] == [False, True, True, False]
    assert trans["scale"] == [1.0] * 4
    assert trans["observation_id"] == obs["id"]


def test_post_ensemble_results(client):
    data = pd.DataFrame([[1, 0.1], [2, 0.2], [3, 0.3], [4, 0.4]], index=[2, 4, 6, 8])
    response_name = "RES"

    # Add priors to ERT config
    builder = ErtConfigBuilder()
    builder.ensemble_size = 2
    builder.add_general_observation("OBS", response_name, data)

    data = [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
    df = pd.DataFrame(data)

    ert = builder.build()

    # Create runpath and run ERT
    run_context = _create_runpath(ert)
    for path in run_context.paths:
        (Path(path) / "poly_0.out").write_text("\n".join([str(nr) for nr in data]))
    ert.load_from_forward_model("default", [True] * len(run_context), 0)

    # Post initial ensemble
    ensemble_id = extraction.post_ensemble_data(ert, builder.ensemble_size)

    # Post ensemble results
    extraction.post_ensemble_results(ert, ensemble_id)

    # Retrieve response data
    data = client.get(f"/ensembles/{ensemble_id}/responses/{response_name}/data")
    stream = io.BytesIO(data.content)
    response_df = pd.read_csv(stream, index_col=0, float_precision="round_trip")
    for realization in range(0, builder.ensemble_size):
        assert_array_equal(response_df.loc[realization].values, df.values.flatten())


def test_post_update_data(client):
    data = pd.DataFrame(np.random.rand(4, 2), index=[2, 4, 6, 8])

    builder = ErtConfigBuilder()
    builder.add_general_observation("OBS", "RES", data)
    ert = builder.build()

    # Post two ensembles
    parent_ensemble_id = extraction.post_ensemble_data(ert, builder.ensemble_size)
    update_id = extraction.post_update_data(ert, parent_ensemble_id, "boruvka")
    child_ensemble_id = extraction.post_ensemble_data(
        ert, builder.ensemble_size, update_id
    )

    # Experiment should have two ensembles
    experiment_id = client.fetch_experiment()
    ensembles = client.get(f"/experiments/{experiment_id}/ensembles").json()
    assert len(ensembles) == 2

    # Parent ensemble should have a child
    assert ensembles[0]["child_ensemble_ids"] == [child_ensemble_id]
    assert ensembles[0]["parent_ensemble_id"] is None

    # Child ensemble should have a parent
    assert ensembles[1]["child_ensemble_ids"] == []
    assert ensembles[1]["parent_ensemble_id"] == parent_ensemble_id


def _make_priors() -> List[Tuple[str, str, dict]]:
    def normal():
        # trans_normal @ clib/lib/enkf/trans_func.cpp
        #
        # Well defined for all real values of a and b
        a, b = random.random(), random.random()
        return (f"NORMAL {a} {b}", dict(function="normal", mean=a, std=b))

    def lognormal():
        # trans_lognormal @ clib/lib/enkf/trans_func.cpp
        #
        # Well defined for all real values of a and b
        a, b = random.random(), random.random()
        return (
            f"LOGNORMAL {a} {b}",
            {"function": "lognormal", "mean": a, "std": b},
        )

    def truncnormal():
        # trans_truncated_normal @ clib/lib/enkf/trans_func.cpp
        #
        # Well defined for all real values of a, b, c and d
        a, b, c, d = [random.random() for _ in range(4)]
        return (
            f"TRUNCATED_NORMAL {a} {b} {c} {d}",
            {
                "function": "ert_truncnormal",
                "mean": a,
                "std": b,
                "min": c,
                "max": d,
            },
        )

    def uniform():
        # trans_unif @ clib/lib/enkf/trans_func.cpp
        #
        # Well defined for all real values of a and b
        a, b = random.random(), random.random()
        return (f"UNIFORM {a} {b}", {"function": "uniform", "min": a, "max": b})

    def loguniform():
        # trans_logunif @ clib/lib/enkf/trans_func.cpp
        #
        # Well defined for strictly positive a, b due to log()
        a, b = random.random() + 1, random.random() + 1  # +1 to avoid zero
        return (
            f"LOGUNIF {a} {b}",
            {"function": "loguniform", "min": a, "max": b},
        )

    def const():
        a = random.random()
        return (f"CONST {a}", {"function": "const", "value": a})

    def duniform():
        # trans_dunif @ clib/lib/enkf/trans_func.cpp
        #
        # Well defined for all real values of b and c, integer values >= 2 of
        # bins (due to division by [bins - 1])
        bins = random.randint(2, 100)
        b, c = random.random(), random.random()
        return (
            f"DUNIF {bins} {b} {c}",
            {"function": "ert_duniform", "bins": bins, "min": b, "max": c},
        )

    def erf():
        # trans_errf @ clib/lib/enkf/trans_func.cpp
        #
        # Well defined for all real values of a, b, c, non-zero real values of d
        # (width) due to division by zero.
        a, b, c, d = [random.random() + 1 for _ in range(4)]  # +1 to all to avoid zero
        return (
            f"ERRF {a} {b} {c} {d}",
            {"function": "ert_erf", "min": a, "max": b, "skewness": c, "width": d},
        )

    def derf():
        # trans_derrf @ clib/lib/enkf/trans_func.cpp
        #
        # Well defined for all real values of a, b, c, non-zero real values of d
        # (width) due to division by zero, integer values >= 2 of bins due to
        # division by (bins - 1)
        bins = random.randint(2, 100)
        a, b, c, d = [random.random() + 1 for _ in range(4)]  # +1 to all to avoid zero
        return (
            f"DERRF {bins} {a} {b} {c} {d}",
            {
                "function": "ert_derf",
                "bins": bins,
                "min": a,
                "max": b,
                "skewness": c,
                "width": d,
            },
        )

    return [
        (_rand_name(), *p())
        for p in (
            normal,
            lognormal,
            truncnormal,
            uniform,
            loguniform,
            const,
            duniform,
            erf,
            derf,
        )
    ]


def _rand_name():
    return "".join(random.choice(string.ascii_lowercase) for _ in range(8))


def _create_runpath(ert: LibresFacade, iteration: int = 0) -> RunContext:
    """
    Instantiate an ERT runpath. This will create the parameter coefficients.
    """
    enkf_main = ert._enkf_main

    run_context = enkf_main.create_ensemble_smoother_run_context(
        iteration=iteration,
        target_filesystem=enkf_main.getEnkfFsManager().getFileSystem("iter"),
    )

    enkf_main.createRunPath(run_context)
    return run_context


def _get_parameters() -> pd.DataFrame:
    params_json = [
        json.loads(path.read_text())
        for path in sorted(Path.cwd().glob("simulations/realization*/coeffs.json"))
    ]

    return pd.DataFrame(params_json)
