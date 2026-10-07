"""The waterflood case: four breakthrough times against 2500 cells.

A field this weakly informed is what localization is for, and these tests check what
the update does to it: that an unlocalized update leaves it worse than the prior, that
either taper repairs that, and that the result predicts report steps nobody
assimilated. One test is an xfail: ert's distance localization crashes when asked for
more than one assimilation.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import resfo

from ert.config import ErtConfig
from ert.mode_definitions import ENSEMBLE_SMOOTHER_MODE, ES_MDA_MODE
from ert.storage import open_storage
from tests.ert.ui_tests.cli.run_cli import run_cli

# The producers, in the row order of `truth_forecast.txt`
PRODUCERS = ("PROD1", "PROD2", "PROD3", "PROD4")


def write_config(name: str, localization: str, realizations: int = 20) -> str:
    """`config.ert` with one localization strategy, and its own storage."""
    lines = []
    for original in Path("config.ert").read_text(encoding="utf-8").splitlines():
        line = original
        if original.startswith("ANALYSIS_SET_VAR PARAMETERS"):
            keyword = original.split()[2]
            strategy = localization if keyword == "FIELD" else "GLOBAL"
            line = f"ANALYSIS_SET_VAR PARAMETERS {keyword} {strategy}"
        elif original.startswith("NUM_REALIZATIONS"):
            line = f"NUM_REALIZATIONS {realizations}"
        lines.append(line)
    lines.extend(
        [
            f"ENSPATH storage-{name}",
            f"RUNPATH simulations-{name}/real-<IENS>/iter-<ITER>",
        ]
    )
    config_file = f"{name}.ert"
    Path(config_file).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_file


def rms(error) -> float:
    return float(np.sqrt((error**2).mean()))


def field_error(config_file: str) -> tuple[float, float]:
    """RMS error of the prior's and the posterior's mean log permeability."""
    truth = resfo.read("truth_permx.bgrdecl")[0][1]
    with open_storage(ErtConfig.from_file(config_file).ens_path, mode="r") as storage:
        experiment = next(iter(storage.experiments))
        ensembles = sorted(experiment.ensembles, key=lambda e: e.iteration)
        errors = []
        for ensemble in (ensembles[0], ensembles[-1]):
            estimate = np.asarray(
                ensemble.load_parameters("PERMX")["values"].mean(dim="realizations")
            )
            # The file is flat and in Fortran order; the field is (nx, ny, nz)
            truth_field = np.log(truth).reshape(estimate.shape, order="F")
            errors.append(rms(estimate - truth_field))
    return errors[0], errors[1]


def forecast_error(config_file: str) -> tuple[float, float]:
    """RMS error of the prior's and the posterior's water cut, after the history."""
    truth = np.loadtxt("truth_forecast.txt")
    with open_storage(ErtConfig.from_file(config_file).ens_path, mode="r") as storage:
        experiment = next(iter(storage.experiments))
        ensembles = sorted(experiment.ensembles, key=lambda e: e.iteration)
        # The forecast is whatever was simulated after the last observation
        last_observed = experiment.observations["summary"]["time"].max()
        errors = []
        for ensemble in (ensembles[0], ensembles[-1]):
            responses = ensemble.load_responses(
                "summary", tuple(ensemble.get_realization_list_with_responses())
            )
            after = responses.filter(pl.col("time") > last_observed)
            predicted = np.array(
                [
                    after.filter(pl.col("response_key") == f"WWCT:{producer}")
                    .group_by("time")
                    .mean()
                    .sort("time")["values"]
                    .to_numpy()
                    for producer in PRODUCERS
                ]
            )
            assert predicted.shape == truth.shape
            errors.append(rms(predicted - truth))
    return errors[0], errors[1]


@pytest.mark.timeout(1200)
@pytest.mark.usefixtures("copy_waterflood")
@pytest.mark.slow
def test_that_the_unlocalized_update_leaves_the_field_further_from_the_truth():
    """Twenty realizations against 2500 cells is the regime localization exists for.

    That localization repairs this is asserted by the next test, which is where the
    two tapers are compared.
    """
    unlocalized = write_config("global", "GLOBAL")
    run_cli(ENSEMBLE_SMOOTHER_MODE, "--disable-monitoring", unlocalized)
    prior_error, unlocalized_error = field_error(unlocalized)
    assert unlocalized_error > prior_error


@pytest.mark.timeout(1200)
@pytest.mark.usefixtures("copy_waterflood")
@pytest.mark.slow
def test_that_both_localization_strategies_repair_the_unlocalized_update():
    """Which taper is *better* is not asserted: tuned, they are hard to separate on
    this case (0.778 adaptive against 0.779 distance at 50 realizations), and at
    20 the radius wins comfortably (0.840 against 0.945) because the threshold in
    `config.ert` is tuned for the larger ensemble -- an adaptive threshold has to
    scale with the ~1/sqrt(N) noise floor of a correlation, a radius does not.
    Ref the README.
    """
    unlocalized = write_config("global", "GLOBAL")
    run_cli(ENSEMBLE_SMOOTHER_MODE, "--disable-monitoring", unlocalized)
    _, unlocalized_error = field_error(unlocalized)

    localized_errors = {}
    for name, strategy in [("adaptive", "ADAPTIVE"), ("distance", "DISTANCE")]:
        config_file = write_config(name, strategy)
        run_cli(ENSEMBLE_SMOOTHER_MODE, "--disable-monitoring", config_file)
        prior_error, localized_errors[strategy] = field_error(config_file)
        assert localized_errors[strategy] < unlocalized_error, strategy

    # And the better of the two is an improvement on the prior, not merely on the
    # damage. Which one that is depends on the ensemble size, hence the min().
    assert min(localized_errors.values()) < prior_error


@pytest.mark.timeout(1200)
@pytest.mark.usefixtures("copy_waterflood")
@pytest.mark.slow
def test_that_assimilating_the_waterflood_history_improves_the_forecast():
    """The wells are observed over the first 16 of 40 report steps; the truth's water
    cut over the other 24 -- during which a third producer breaks through and a
    fourth stays dry throughout -- ships with the case. Fitting the history is not
    enough to pass.
    """
    config_file = write_config("forecast", "ADAPTIVE")
    run_cli(ES_MDA_MODE, "--disable-monitoring", config_file)

    prior, posterior = forecast_error(config_file)
    assert posterior < 0.5 * prior


@pytest.mark.timeout(1200)
@pytest.mark.usefixtures("copy_waterflood")
@pytest.mark.slow
@pytest.mark.xfail(
    reason="DISTANCE localization crashes under ES-MDA: the taper and the assimilated "
    "observations disagree in number once some are deactivated",
    raises=Exception,
    strict=False,
)
def test_that_distance_localization_survives_several_assimilations():
    """ES-MDA with DISTANCE dies in `_distance.py` at `K *= rho`.

    On this case: ValueError, operands could not be broadcast together with shapes
    (2500, 31) (2500, 33) (2500, 31), at the third assimilation, with all 36
    observations carrying positions. ES (one assimilation) is unaffected.
    """
    config_file = write_config("mda-distance", "DISTANCE")
    run_cli(ES_MDA_MODE, "--disable-monitoring", config_file)
