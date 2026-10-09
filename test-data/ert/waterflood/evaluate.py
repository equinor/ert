#!/usr/bin/env python3
"""Score a run of this case -- on the parameters and the forecast, not only the misfit.

Run it from this directory, after `ert es_mda config.ert`:

    python evaluate.py

For each ensemble it reports

- the mean squared normalized data misfit, over the observations that were assimilated;
- the RMS error, the correlation with the truth, and the remaining ensemble spread,
  of the ensemble mean of log permeability -- the truth being what
  `generate_files.py` wrote to `truth_permx.bgrdecl`;
- the RMS error of the ensemble mean water cut over the report steps *after* the
  history, against `truth_forecast.txt`.

The last of these is the point of the exercise, and is what a case with a known truth
can score that a misfit cannot: an ensemble can always be made to fit the history it
was shown.
"""

import datetime

import numpy as np
import polars as pl
import resfo
from definition import dt, forecast_steps, nx, ny, producers, start_date

from ert.config import ErtConfig
from ert.storage import open_storage


def truth_log_permeability() -> np.ndarray:
    """`(nx, ny)` log permeability of the truth, as ert stores it (`INIT_TRANSFORM`)."""
    permx = resfo.read("truth_permx.bgrdecl")[0][1].reshape(nx, ny, order="F")
    return np.log(permx)


def forecast_dates() -> list[datetime.datetime]:
    start = datetime.datetime.fromisoformat(start_date)
    return [
        start + datetime.timedelta(days=float(step * dt)) for step in forecast_steps
    ]


def misfit(ensemble, observations: pl.DataFrame) -> float:
    """Mean squared normalized data misfit, over realizations and observations."""
    responses = ensemble.load_responses(
        "summary", tuple(ensemble.get_realization_list_with_responses())
    )
    joined = observations.join(responses, on=["response_key", "time"], how="inner")
    residual = (joined["values"] - joined["observations"]) / joined["std"]
    return float((residual**2).mean())


def forecast_error(ensemble, truth: np.ndarray) -> float:
    """RMS error of the ensemble mean water cut, over the forecast steps."""
    responses = ensemble.load_responses(
        "summary", tuple(ensemble.get_realization_list_with_responses())
    )
    predicted = np.array(
        [
            [
                responses.filter(
                    (pl.col("response_key") == f"WWCT:{well.name}")
                    & (pl.col("time") == date)
                )["values"].mean()
                for date in forecast_dates()
            ]
            for well in producers
        ]
    )
    return float(np.sqrt(((predicted - truth) ** 2).mean()))


def parameter_scores(ensemble, truth: np.ndarray) -> tuple[float, float, float]:
    """RMS error, correlation with the truth, and spread, of log permeability."""
    values = ensemble.load_parameters("PERMX")["values"]
    estimate = np.asarray(values.mean(dim="realizations")).reshape(nx, ny)
    spread = np.asarray(values.std(dim="realizations")).mean()
    return (
        float(np.sqrt(((estimate - truth) ** 2).mean())),
        float(np.corrcoef(estimate.ravel(), truth.ravel())[0, 1]),
        float(spread),
    )


if __name__ == "__main__":
    config = ErtConfig.from_file("config.ert")
    truth_water_cut = np.loadtxt("truth_forecast.txt")
    truth_log_perm = truth_log_permeability()

    with open_storage(config.ens_path, mode="r") as storage:
        experiment = max(storage.experiments, key=lambda e: e.name)
        observations = experiment.observations["summary"]

        header = (
            "ensemble",
            "misfit",
            "RMSE logK",
            "corr logK",
            "spread logK",
            "RMSE forecast",
        )
        print("{:>12} {:>8} {:>10} {:>10} {:>12} {:>14}".format(*header))
        for ensemble in sorted(experiment.ensembles, key=lambda e: e.iteration):
            error, correlation, spread = parameter_scores(ensemble, truth_log_perm)
            print(
                f"{ensemble.name:>12} {misfit(ensemble, observations):8.1f} "
                f"{error:10.3f} {correlation:10.3f} {spread:12.3f} "
                f"{forecast_error(ensemble, truth_water_cut):14.3f}"
            )
