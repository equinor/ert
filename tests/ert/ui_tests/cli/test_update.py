from __future__ import annotations

import os
import stat
import warnings
from pathlib import Path

import hypothesis.strategies as st
import numpy as np
from hypothesis import given, note, settings
from pytest import MonkeyPatch, TempPathFactory

from ert.cli.main import ErtCliError
from ert.mode_definitions import ENSEMBLE_SMOOTHER_MODE
from ert.storage import open_storage

from .run_cli import run_cli

names = st.text(
    min_size=1,
    max_size=8,
    alphabet=st.characters(
        min_codepoint=ord("!"),
        max_codepoint=ord("~"),
        exclude_characters="\"'$,:%",  # These have specific meaning in configs
    ),
)

DISTRIBUTION_PARAMETERS: dict[str, list[str]] = {
    "NORMAL": ["MEAN", "STD"],
    "LOGNORMAL": ["MEAN", "STD"],
    "TRUNCATED_NORMAL": ["MEAN", "STD", "MIN", "MAX"],
    "TRIANGULAR": ["MIN", "MODE", "MAX"],
    "UNIFORM": ["MIN", "MAX"],
    "DUNIF": ["STEPS", "MIN", "MAX"],
    "ERRF": ["MIN", "MAX", "SKEWNESS", "WIDTH"],
    "DERRF": ["STEPS", "MIN", "MAX", "SKEWNESS", "WIDTH"],
    "LOGUNIF": ["MIN", "MAX"],
    "CONST": ["VALUE"],
    "RAW": [],
}


@st.composite
def distribution_values(draw, k, vs):
    d = {}
    biggest = 100.0
    if "LOG" in k:
        biggest = 10.0
    epsilon = biggest / 1000.0
    if "MIN" in vs:
        d["MIN"] = draw(st.floats(min_value=epsilon, max_value=biggest / 10.0))
    if "MAX" in vs:
        d["MAX"] = draw(st.floats(min_value=d["MIN"] + 5 * epsilon, max_value=biggest))
    if "MEAN" in vs:
        d["MEAN"] = draw(
            st.floats(
                min_value=d.get("MIN", 2 * epsilon) + epsilon,
                max_value=d.get("MAX", biggest) - epsilon,
            )
        )
    if "MODE" in vs:
        d["MODE"] = draw(
            st.floats(
                min_value=d.get("MIN", 2 * epsilon) + epsilon,
                max_value=d.get("MAX", biggest) - epsilon,
            )
        )
    if "STEPS" in vs:
        d["STEPS"] = draw(st.integers(min_value=2, max_value=10))
    return [d.get(v, draw(st.floats(min_value=0.1, max_value=1.0))) for v in vs]


distributions = st.one_of(
    [
        st.tuples(
            st.just(k),
            distribution_values(k, vs),
        )
        for k, vs in DISTRIBUTION_PARAMETERS.items()
    ]
)

config_contents = """\
NUM_REALIZATIONS {num_realizations}
QUEUE_SYSTEM LOCAL
QUEUE_OPTION LOCAL MAX_RUNNING {num_realizations}
ENSPATH storage
RANDOM_SEED 1234

OBS_CONFIG observations
GEN_KW COEFFS coeff_priors
GEN_DATA POLY_RES RESULT_FILE:poly.out

INSTALL_JOB poly_eval POLY_EVAL
FORWARD_MODEL poly_eval

ANALYSIS_SET_VAR OBSERVATIONS AUTO_SCALE *
"""

observation = """\
GENERAL_OBSERVATION POLY_OBS_{i} {{
        DATA       = POLY_RES;
        INDEX_FILE = index_{i}.txt;
        OBS_FILE   = poly_obs_{i}.txt;
}};
"""

poly_eval = """\
#!/usr/bin/env python3
import json
import numpy as np
coeffs = json.load(open("parameters.json"))["COEFFS"]
c = [np.array(coeffs[f"coeff_" + str(i)]) for i in range(len(coeffs))]
with open("poly.out", "w", encoding="utf-8") as f:
    f.write("\\n".join(map(str, [np.polyval(c, x) for x in range({num_points})])))
"""

POLY_EVAL = "EXECUTABLE poly_eval.py"


@settings(max_examples=3)
@given(
    num_realizations=st.integers(min_value=20, max_value=40),
    num_points=st.integers(min_value=1, max_value=20),
    distributions=st.lists(distributions, min_size=1, max_size=10),
    data=st.data(),
)
def test_update_lowers_generalized_variance_or_deactivates_observations(
    tmp_path_factory: TempPathFactory,
    num_realizations: int,
    num_points: int,
    distributions: list[tuple[str, list[float]]],
    data,
):
    indices = data.draw(
        st.lists(
            st.integers(min_value=0, max_value=num_points - 1),
            min_size=1,
            max_size=num_points,
            unique=True,
        )
    )
    values = data.draw(
        st.lists(
            st.floats(min_value=-10.0, max_value=10.0),
            min_size=len(indices),
            max_size=len(indices),
        )
    )
    errs = data.draw(
        st.lists(
            st.floats(min_value=0.1, max_value=0.5),
            min_size=len(indices),
            max_size=len(indices),
        )
    )
    num_groups = data.draw(st.integers(min_value=1, max_value=num_points))
    per_group = num_points // num_groups

    tmp_path = tmp_path_factory.mktemp("parameter_example")
    note(f"Running in directory {tmp_path}")
    with MonkeyPatch.context() as patch:
        patch.chdir(tmp_path)
        contents = config_contents.format(
            num_realizations=num_realizations,
        )
        note(f"config file: {contents}")
        Path("config.ert").write_text(contents, encoding="utf-8")
        py = Path("poly_eval.py")
        py.write_text(poly_eval.format(num_points=num_points), encoding="utf-8")
        mode = os.stat(py)
        os.chmod(py, mode.st_mode | stat.S_IEXEC)

        for i in range(num_groups):
            with open("observations", mode="a", encoding="utf-8") as f:
                f.write(observation.format(i=i))
            Path(f"poly_obs_{i}.txt").write_text(
                "\n".join(
                    f"{x} {y}"
                    for x, y in zip(
                        values[i * per_group : (i + 1) * per_group],
                        errs[i * per_group : (i + 1) * per_group],
                        strict=False,
                    )
                ),
                encoding="utf-8",
            )
            Path(f"index_{i}.txt").write_text(
                "\n".join(f"{x}" for x in indices[i * per_group : (i + 1) * per_group]),
                encoding="utf-8",
            )

        Path("coeff_priors").write_text(
            "\n".join(
                f"coeff_{i} {d} {' '.join(str(p) for p in v)}"
                for i, (d, v) in enumerate(distributions)
            ),
            encoding="utf-8",
        )
        Path("POLY_EVAL").write_text(POLY_EVAL, encoding="utf-8")

        success = True
        with warnings.catch_warnings(record=True) as all_warnings:
            warnings.simplefilter("always")
            try:
                run_cli(
                    ENSEMBLE_SMOOTHER_MODE,
                    "--disable-monitoring",
                    "--experiment-name",
                    "experiment",
                    "config.ert",
                )
            except ErtCliError as err:
                success = False
                se = str(err)
                # "No active observations" is expected
                # when std deviation is too low, the
                # other errors are discussed here:
                # https://github.com/equinor/ert/issues/9581
                # https://github.com/equinor/ert/issues/9585
                assert (
                    "No active observations" in se
                    or "Matrix is singular" in se
                    or "math domain error" in se
                    or "math range error" in se
                )

        if any("Ill-conditioned matrix" not in str(w.message) for w in all_warnings):
            success = False

        if success:
            with open_storage("storage") as storage:
                experiment = storage.get_experiment_by_name("experiment")
                prior = experiment.get_ensemble_by_name("iter-0").load_scalars()
                posterior = experiment.get_ensemble_by_name("iter-1").load_scalars()

            assert (
                np.linalg.det(np.cov(posterior.to_numpy(), rowvar=False))
                <= np.linalg.det(np.cov(prior.to_numpy(), rowvar=False)) + 0.001
            )
