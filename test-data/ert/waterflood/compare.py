#!/usr/bin/env python3
"""Run several update schemes on this case, and compare what they do to it.

    python compare.py [nRealizations]

Each run is the same case -- same prior, same truth, same observations, same seed --
updated by a different scheme: with and without iterations (ES and ES-MDA, plus EnIF),
and with and without localization of the permeability field (`GLOBAL`, `ADAPTIVE` from
the ensemble's own correlations, `DISTANCE` from the positions the observations carry).
The two scalar parameters are left global throughout, so that the rows differ in the
treatment of the field alone.

Each is then scored on the three things `evaluate.py` scores, the last of which --
the error of the prediction over the report steps that were never assimilated -- is
the one that cannot be bought by fitting harder.

The runs are independent, so this is minutes of wall time: it is meant to be run
manually, rather than automatically as part of the test suite.
`tests/ert/ui_tests/cli/test_waterflood.py` is what runs automatically, asserting the
properties that hold whatever the numbers, on a smaller ensemble.
"""

import re
import subprocess
import sys
from pathlib import Path
from uuid import UUID

import numpy as np
from definition import nsteps_history
from evaluate import (
    forecast_error,
    misfit,
    parameter_scores,
    truth_log_permeability,
)

from ert.config import ErtConfig
from ert.storage import open_storage

# (label, ert mode, localization). EnIF brings its own sparse-precision structure,
# so it is listed once, at its default.
# (label, ert mode, field localization, tuning). The tuning is the correlation
# threshold that ADAPTIVE cuts at, or the radius (in metres) that DISTANCE tapers
# over -- the latter living on the observations, so it is written into a copy of them.
RUNS = [
    ("ES, no localization", "ensemble_smoother", "GLOBAL", {}),
    ("ES, adaptive", "ensemble_smoother", "ADAPTIVE", {}),
    ("ES, distance", "ensemble_smoother", "DISTANCE", {}),
    ("ES-MDA, no localization", "es_mda", "GLOBAL", {}),
    ("ES-MDA, adaptive", "es_mda", "ADAPTIVE", {}),
    ("ES-MDA, distance", "es_mda", "DISTANCE", {}),
    ("EnIF", "ensemble_information_filter", None, {}),
]

# The tuning the README's sweeps used, e.g.
#   ("ES, adaptive 0.5", "ensemble_smoother", "ADAPTIVE", {"threshold": 0.5}),
#   ("ES, distance 250 m", "ensemble_smoother", "DISTANCE", {"radius": 250}),


def write_observations(tag: str, radius: float) -> str:
    """A copy of the observations, tapered over `radius` instead of their own."""
    path = f"observations-{tag}.txt"
    text = Path("observations.txt").read_text(encoding="utf-8")
    Path(path).write_text(
        re.sub(r"RADIUS = [\d.]+", f"RADIUS = {radius}", text), encoding="utf-8"
    )
    return path


def write_config(
    tag: str, localization: str | None, realizations: int, tuning: dict
) -> Path:
    """`config.ert` with one localization strategy and its own runpath.

    The storage is left at its default (`ENSPATH`), shared by every run below, so
    that each can be told apart only by its own experiment and ensemble names.
    """
    lines = []
    for original in Path("config.ert").read_text(encoding="utf-8").splitlines():
        line = original
        if original.startswith("ANALYSIS_SET_VAR PARAMETERS") and localization:
            # Only the field's strategy varies; the two scalars are left global, so
            # that what the rows differ in is the localization of the field alone.
            # (DISTANCE is a field-only strategy in any case.)
            keyword = original.split()[2]
            strategy = localization if keyword == "FIELD" else "GLOBAL"
            line = f"ANALYSIS_SET_VAR PARAMETERS {keyword} {strategy}"
        elif original.startswith("NUM_REALIZATIONS"):
            line = f"NUM_REALIZATIONS {realizations}"
        elif "LOCALIZATION_CORRELATION_THRESHOLD" in original and "threshold" in tuning:
            line = (
                "ANALYSIS_SET_VAR STD_ENKF LOCALIZATION_CORRELATION_THRESHOLD "
                f"{tuning['threshold']}"
            )
        elif original.startswith("OBS_CONFIG") and "radius" in tuning:
            line = f"OBS_CONFIG {write_observations(tag, tuning['radius'])}"
        lines.append(line)
    lines.append(f"RUNPATH simulations-{tag}/real-<IENS>/iter-<ITER>")
    path = Path(f"compare-{tag}.ert")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def experiment_ids(ens_path: Path) -> set[UUID]:
    """The experiments already in storage, to spot the one a run just added."""
    if not ens_path.exists():
        return set()
    with open_storage(ens_path, mode="r") as storage:
        return {experiment.id for experiment in storage.experiments}


def score(ens_path: Path, experiment_id: UUID) -> dict[str, tuple[float, ...]]:
    """The prior's and the last posterior's scores, as `evaluate.py` computes them."""
    truth_water_cut = np.loadtxt("truth_forecast.txt")
    truth_log_perm = truth_log_permeability()

    out = {}
    with open_storage(ens_path, mode="r") as storage:
        experiment = storage.get_experiment(experiment_id)
        observations = experiment.observations
        ensembles = sorted(experiment.ensembles, key=lambda e: e.iteration)
        for label, ensemble in [("prior", ensembles[0]), ("posterior", ensembles[-1])]:
            error, correlation, spread = parameter_scores(ensemble, truth_log_perm)
            out[label] = (
                misfit(ensemble, observations),
                error,
                correlation,
                spread,
                forecast_error(ensemble, truth_water_cut),
            )
    return out


if __name__ == "__main__":
    realizations = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    print(
        f"{realizations} realizations, assimilating the first {nsteps_history} "
        "report steps\n"
    )
    header = ("", "misfit", "RMSE logK", "corr logK", "spread", "RMSE forecast")
    print("{:>24} {:>8} {:>10} {:>10} {:>8} {:>14}".format(*header))

    prior_printed = False
    for label, mode, localization, tuning in RUNS:
        tag = f"{label.replace(', ', '-').replace(' ', '').lower()}-{realizations}"
        config_file = write_config(tag, localization, realizations, tuning)
        ens_path = Path(ErtConfig.from_file(str(config_file)).ens_path)

        before = experiment_ids(ens_path)
        result = subprocess.run(
            [
                "ert",
                mode,
                "--disable-monitoring",
                "--target-ensemble",
                f"{tag}-%d",
                "--experiment-name",
                tag,
                str(config_file),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            print(
                f"{label:>24}   failed: {result.stderr.strip().splitlines()[-1][:60]}"
            )
            config_file.unlink()
            Path(f"observations-{tag}.txt").unlink(missing_ok=True)
            continue

        (experiment_id,) = experiment_ids(ens_path) - before
        scores = score(ens_path, experiment_id)
        if not prior_printed:
            print(
                "{:>24} {:8.1f} {:10.3f} {:10.3f} {:8.3f} {:14.3f}".format(
                    "prior", *scores["prior"]
                )
            )
            prior_printed = True
        print(
            "{:>24} {:8.1f} {:10.3f} {:10.3f} {:8.3f} {:14.3f}".format(
                label, *scores["posterior"]
            )
        )

        config_file.unlink()
        Path(f"observations-{tag}.txt").unlink(missing_ok=True)
