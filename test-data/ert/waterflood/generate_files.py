"""Contains code that was used to generate files expected by ert.

Run it from this directory to regenerate `CASE.EGRID`, `observations.txt`,
`truth_permx.bgrdecl` and `truth_forecast.txt`:

    python generate_files.py

The observations are the response of one realization of the model -- the truth --
perturbed by noise of the standard deviation that is reported as their error. The
truth's permeability, and the water cut it goes on to produce over the report steps
that are *not* observed, are written out too, so that the update can be scored on the
parameters and on the forecast, and not only on the data misfit (ref `evaluate.py`).
"""

import datetime
from pathlib import Path

import numpy as np
import resfo
import xtgeo
from definition import (
    breakthrough_steps,
    dt,
    dx,
    dz,
    forecast_steps,
    injectors,
    localization_radius,
    nsteps,
    nx,
    ny,
    obs_steps,
    producers,
    start_date,
    truth_corr_length,
    truth_seed,
    truth_sor,
)

from waterflood import make_model, sample_prior_permeability, simulate

# Relative and absolute parts of the observation error of each summary keyword,
# and the range that the perturbed value is confined to, where it is physical
OBS_ERROR = {
    "WWCT": (0.05, 0.02),  # water cut [-]
    "WOPR": (0.10, 5.0),  # oil rate [m3/day]
    "WBHP": (0.01, 1.0),  # bottom-hole pressure [bar]
}
OBS_RANGE = {"WWCT": (0.0, 1.0), "WOPR": (0.0, None), "WBHP": (None, None)}

# Producers observed as a water cut breakthrough instead of a water cut summary
# observation, and the water cut threshold each is observed to cross. PROD1
# never waters, so it stays a summary observation.
BREAKTHROUGH_THRESHOLDS = {
    "PROD2": 0.2,
    "PROD3": 0.01,
    "PROD4": 0.5,
}
BREAKTHROUGH_ERROR = 30.0  # uncertainty of the observed breakthrough date [days]


def create_egrid_file():
    """A box grid of one layer, in metres. Its cell centres locate the observations."""
    grid = xtgeo.create_box_grid(
        dimension=(nx, ny, 1), increment=(dx, dx, dz), origin=(0.0, 0.0, 0.0)
    )
    grid.to_file("CASE.EGRID", "egrid")


def find_breakthrough_date(
    values: list[float], threshold: float, start: datetime.date
) -> datetime.date:
    """The truth's first date at which `values` crosses `threshold`.

    Uses `breakthrough_steps`, every report step of the whole simulation, rather
    than the quarterly `obs_steps`, so that a crossing between, or after, the
    quarterly steps is not missed.
    """
    step = next(step for step in breakthrough_steps if values[step - 1] >= threshold)
    return start + datetime.timedelta(days=float(step * dt))


def create_observations(vectors: dict[str, list[float]], rng: np.random.Generator):
    """Perturb the truth, and write it as bulk summary and breakthrough
    observations at the wells.

    The summary observations are written to a csv file referenced by a single
    SUMMARY configuration, with the position of each well given once, in a
    WELL block, so that the case can be run with distance based localization
    as well as with adaptive localization. Wells in `BREAKTHROUGH_THRESHOLDS`
    get a BREAKTHROUGH observation instead of a water cut summary observation,
    inheriting the same well's LOCALIZATION.
    """
    positions = {well.name: well for well in [*injectors, *producers]}
    start = datetime.date.fromisoformat(start_date)

    with Path("observations.csv").open("w", encoding="utf-8") as f:
        f.write("well,keyword,value,error,date\n")
        for step in obs_steps:
            date = start + datetime.timedelta(days=float(step * dt))
            for key, values in vectors.items():
                keyword, well_name = key.split(":")
                if keyword == "WWCT" and well_name in BREAKTHROUGH_THRESHOLDS:
                    continue
                relative, absolute = OBS_ERROR[keyword]
                truth = values[step - 1]
                error = max(relative * abs(truth), absolute)
                # A water cut outside [0, 1] would be unphysical, so the noise is
                # confined to what the measurement could have reported
                value = np.clip(
                    truth + rng.normal(loc=0.0, scale=error), *OBS_RANGE[keyword]
                )

                f.write(
                    f"{well_name},{keyword},{value:.16e},{error:.16e},{date:%Y-%m-%d}\n"
                )

    breakthrough_dates = {
        well_name: find_breakthrough_date(
            vectors[f"WWCT:{well_name}"], threshold, start
        )
        + datetime.timedelta(days=rng.normal(loc=0.0, scale=BREAKTHROUGH_ERROR))
        for well_name, threshold in BREAKTHROUGH_THRESHOLDS.items()
    }

    with Path("observations.txt").open("w", encoding="utf-8") as f:
        f.write(
            """SUMMARY {
    VALUES = observations.csv;
"""
        )
        for well_name, well in positions.items():
            f.write(
                f"""    WELL {well_name} {{
        LOCALIZATION {{
            EAST = {well.east};
            NORTH = {well.north};
            RADIUS = {localization_radius};
        }};
"""
            )
            if well_name in BREAKTHROUGH_THRESHOLDS:
                f.write(
                    f"""        BREAKTHROUGH {{
            KEY       = WWCT;
            THRESHOLD = {BREAKTHROUGH_THRESHOLDS[well_name]};
            DATE      = {breakthrough_dates[well_name]:%Y-%m-%d};
            ERROR     = {BREAKTHROUGH_ERROR};
        }};
"""
                )
            f.write("    };\n")
        f.write("};\n")


if __name__ == "__main__":
    create_egrid_file()

    rng = np.random.default_rng(truth_seed)
    permx = sample_prior_permeability(truth_corr_length, rng)
    vectors = simulate(make_model(permx, truth_sor))

    resfo.write(
        "truth_permx.bgrdecl",
        [("PERMX   ", permx.flatten(order="F").astype(np.float32))],
    )
    np.savetxt(
        "truth_forecast.txt",
        [
            [vectors[f"WWCT:{well.name}"][step - 1] for step in forecast_steps]
            for well in producers
        ],
        header="Water cut of the truth over the report steps after the history, "
        f"{len(producers)} producers (rows) x {len(forecast_steps)} steps (columns)",
    )

    water_cut = [vectors[f"WWCT:{well.name}"][nsteps - 1] for well in producers]
    print(
        f"Truth: permeability {permx.min():.0f}-{permx.max():.0f} mD, "
        f"final water cut {np.round(water_cut, 2)}"
    )

    create_observations(vectors, rng)
