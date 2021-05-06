import io
import pytest
import json
from typing import List, Tuple
from random import random, randint
from pathlib import Path

import pandas as pd
from numpy.testing import assert_almost_equal

from ecl.util.util import BoolVector
from res.enkf.res_config import ResConfig
from res.enkf.enkf_main import EnKFMain
from res.enkf import ErtRunContext
from ert_shared.storage import extraction
from ert_shared.libres_facade import LibresFacade


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

    def add_prior(self, name, entry):
        assert name not in self._priors
        self._priors[name] = entry
        return self

    def build(self, path=None):
        if path is None:
            path = Path.cwd()
        (path / "JOB").write_text("EXECUTABLE /usr/bin/true\n")

        self._build_ert(path)
        self._build_priors(path)

        config = ResConfig(str(path / "test.ert"))
        enkfmain = EnKFMain(config)

        # The C code doesn't do resource counting correctly, so we need to hook
        # ResConfig to EnKFMain because otherwise ResConfig will be deleted and
        # EnKFMain will use a dangling pointer.
        enkfmain.__config = config

        return LibresFacade(enkfmain)

    def _build_ert(self, path):
        f = (path / "test.ert").open("w")

        # Default
        f.write(
            "JOBNAME poly_%d\n"
            f"NUM_REALIZATIONS {self.ensemble_size}\n"
            "INSTALL_JOB job JOB\n"
            "SIMULATION_JOB job\n"
        )

    def _build_priors(self, path):
        if not self._priors:
            return

        with (path / "test.ert").open("a") as f:
            f.write("GEN_KW COEFFS coeffs.json.in coeffs.json coeffs_priors\n")
        with (path / "coeffs.json.in").open("w") as f:
            f.write("{\n")
            f.write(",\n".join(f'  "{name}": <{name}>' for name in self._priors))
            f.write("}\n")
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

    id = client.fetch_experiment()

    # Name is "default"
    for ens in client.get(f"/experiments/{id}/ensembles").json():
        assert (
            client.get(f"/ensembles/{ens['id']}/metadata").json()["name"] == "default"
        )

    # No priors exist
    assert client.get(f"/experiments/{id}/priors").json() == {}


def test_empty_ensemble_with_name(client):
    name = _rand_name()

    # Create case with given name
    ert = ErtConfigBuilder().build()
    ert.select_or_create_new_case(name)

    # Post initial ensemble
    extraction.post_ensemble_data(ert, -1)

    # Compare results
    id = client.fetch_experiment()
    for ens in client.get(f"/experiments/{id}/ensembles").json():
        assert client.get(f"/ensembles/{ens['id']}/metadata").json()["name"] == name


def test_priors(client):
    priors = _make_priors()

    # Add priors to ERT config
    builder = ErtConfigBuilder()
    for name, conf, _ in priors:
        builder.add_prior(name, conf)
    ert = builder.build()

    # Post initial ensemble
    extraction.post_ensemble_data(ert, -1)

    # Compare results
    id = client.fetch_experiment()
    actual_priors = client.get(f"/experiments/{id}/priors").json()
    assert len(priors) == len(actual_priors)
    for name, _, resp in priors:
        assert actual_priors[f"COEFFS:{name}"] == resp


def test_parameters(client):
    priors = _make_priors()

    # Add priors to ERT config
    builder = ErtConfigBuilder()
    builder.ensemble_size = 10  # randint(5, 20)
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
    assert len(parameters) == len(priors) + 2
    for name, _, prior in priors:
        assert f"COEFFS:{name}" in parameters
        if prior["function"] in ("lognormal", "loguniform"):
            assert f"LOG10_COEFFS:{name}" in parameters

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
            headers={"accept": "application/x-dataframe"},
        ).content
        stream = io.BytesIO(record_data)
        df = pd.read_csv(stream, index_col=0, float_precision="round_trip")

        # ERT produces a low-quality version
        assert_almost_equal(parameters_df[col].values, df.values.flatten(), decimal=4)


def _make_priors() -> List[Tuple[str, str, dict]]:
    def normal():
        a, b = random(), random()
        return (f"NORMAL {a} {b}", dict(function="normal", mean=a, std=b))

    def lognormal():
        a, b = random(), random()
        return (
            f"LOGNORMAL {a} {b}",
            {"function": "lognormal", "mean": a, "std": b},
        )

    def truncnormal():
        a, b, c, d = [random() for _ in range(4)]
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
        a, b = random(), random()
        return (f"UNIFORM {a} {b}", {"function": "uniform", "min": a, "max": b})

    def loguniform():
        a, b = random(), random()
        return (
            f"LOGUNIF {a} {b}",
            {"function": "loguniform", "min": a, "max": b},
        )

    def const():
        a = random()
        return (f"CONST {a}", {"function": "const", "value": a})

    def duniform():
        bins = randint(1, 100)
        b, c = random(), random()
        return (
            f"DUNIF {bins} {b} {c}",
            {"function": "ert_duniform", "bins": bins, "min": b, "max": c},
        )

    def erf():
        a, b, c, d = [random() for _ in range(4)]
        return (
            f"ERRF {a} {b} {c} {d}",
            {"function": "ert_erf", "min": a, "max": b, "skewness": c, "width": d},
        )

    def derf():
        bins = randint(1, 100)
        a, b, c, d = [random() for _ in range(4)]
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
    import random, string

    return "".join(random.choice(string.ascii_lowercase) for _ in range(8))


def _create_runpath(ert: LibresFacade):
    """
    Instantiate an ERT runpath. This will create the parameter coefficients.
    """
    enkf_main = ert._enkf_main
    result_fs = ert.get_current_fs()

    model_config = enkf_main.getModelConfig()
    runpath_fmt = model_config.getRunpathFormat()
    jobname_fmt = model_config.getJobnameFormat()
    subst_list = enkf_main.getDataKW()

    run_context = ErtRunContext.ensemble_experiment(
        result_fs,
        BoolVector(default_value=True, initial_size=ert.get_ensemble_size()),
        runpath_fmt,
        jobname_fmt,
        subst_list,
        0,
    )

    ert._enkf_main.getEnkfSimulationRunner().createRunPath(run_context)


def _get_parameters() -> pd.DataFrame:
    params_json = [
        json.loads(path.read_text())
        for path in sorted(Path.cwd().glob("simulations/realization*/coeffs.json"))
    ]

    return pd.DataFrame(params_json)
