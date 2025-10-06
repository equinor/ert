"""
Contains code that was used to generate files expected by ert.
"""

from collections.abc import Callable
from pathlib import Path
from textwrap import dedent

import numpy as np
import numpy.typing as npt
import pandas as pd
import resfo
import xtgeo
from definition import Coordinate, obs_coordinates, obs_times
from heat_equation import heat_equation, sample_prior_conductivity

# Some seeds produce priors that yield poor results.
# Worth playing around with.
rng = np.random.default_rng(1234)


def create_egrid_file():
    NCOL = 10
    NROW = 10
    NLAY = 1
    grid = xtgeo.create_box_grid(dimension=(NCOL, NROW, NLAY))
    grid.to_file("CASE.EGRID", "egrid")


def make_observations(
    coordinates: list[Coordinate],
    times: npt.NDArray[np.int_],
    field: npt.NDArray[np.float64],
    error: Callable,
) -> pd.DataFrame:
    """Generate synthetic observations by adding noise to true field-values.
    Parameters
    ----------
    error: Callable
        Function that takes a single argument (the true field value) and returns
        a value to be used as the standard deviation of the noise.
    """
    d = pd.DataFrame(
        {
            "k": pd.Series(dtype=int),
            "x": pd.Series(dtype=int),
            "y": pd.Series(dtype=int),
            "value": pd.Series(dtype=float),
            "sd": pd.Series(dtype=float),
        }
    )

    # Create dataframe with observations and necessary meta data.
    for coordinate in coordinates:
        for k in times:
            # The reason for u[k, y, x] instead of the perhaps more natural u[k, x, y],
            # is due to a convention followed by matplotlib's `pcolormesh`
            # See documentation for details.
            value = field[k, coordinate.x, coordinate.y]
            sd = error(value)
            df_ = pd.DataFrame(
                {
                    "k": [k],
                    "x": [coordinate.x],
                    "y": [coordinate.y],
                    "value": [value + rng.normal(loc=0.0, scale=sd)],
                    "sd": [sd],
                }
            )
            d = pd.concat([d, df_])
    d = d.set_index(["k", "x", "y"], verify_integrity=True)

    return d


def generate_priors():
    """Generates and saves 10 random conductivity field realizations.

    Uses a prior sampling function to create conductivity fields,
    then saves each field to a separate .bgrdecl file in ECLIPSE format
    using Fortran-style ordering. Used for testing when FORWARD_INIT
    is disabled.
    """
    corr_lengths = rng.normal(loc=0.8, scale=0.1, size=10)
    for i in range(10):
        cond = sample_prior_conductivity(
            ensemble_size=1, nx=nx, rng=rng, corr_length=corr_lengths[i]
        )
        resfo.write(
            f"cond_{i}.bgrdecl",
            [("COND    ", cond.flatten(order="F").astype(np.float32))],
        )


if __name__ == "__main__":
    create_egrid_file()

    # Number of grid-cells in x and y direction
    nx = 10

    # time steps
    k_start = 0
    k_end = 500

    # Define initial condition, i.e., the initial temperature distribution.
    # How you define initial conditions will effect the spread of results,
    # i.e., how similar different realisations are.
    u_init = np.zeros((k_end, nx, nx))
    u_init[:, 5:7, 5:7] = 100

    cond_truth = sample_prior_conductivity(
        ensemble_size=1, nx=nx, rng=rng, corr_length=0.8
    ).reshape(nx, nx)

    # Resolution in the x-direction (nothing to worry about really)
    dx = 1

    # Calculate maximum `dt`.
    # If higher values are used, the numerical solution will become unstable.
    # Note that this could be avoided if we used an implicit solver.
    dt = dx**2 / (4 * np.max(cond_truth))

    u_t = heat_equation(u_init, cond_truth, dx, dt, k_start, k_end, rng=rng)

    d = make_observations(
        obs_coordinates, obs_times, u_t, lambda value: abs(0.05 * value)
    )

    with open("observations", "w", encoding="utf-8") as fout:
        fout.writelines(
            dedent(
                f"""
            GENERAL_OBSERVATION MY_OBS_{obs_time} {{
                DATA       = MY_RESPONSE;
                RESTART    = {obs_time};
                OBS_FILE   = obs_{obs_time}.txt;
            }};"""
            )
            for obs_time in obs_times
        )

    for obs_time in obs_times:
        df = d.iloc[d.index.get_level_values("k") == obs_time]
        Path(f"obs_{obs_time}.txt").write_text(
            df.sort_index().to_csv(header=False, index=False, sep=" "), encoding="utf-8"
        )

    generate_priors()
