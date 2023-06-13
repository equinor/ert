import inspect
import os
import stat
from argparse import ArgumentParser
from collections import namedtuple
from datetime import datetime, timedelta
from textwrap import dedent
from typing import Callable, List

import numpy as np
import pytest
import xtgeo

from ert.__main__ import ert_parser
from ert._c_wrappers.enkf import EnKFMain, ErtConfig
from ert.cli import ENSEMBLE_SMOOTHER_MODE
from ert.cli.main import run_cli
from ert.storage import open_storage


@pytest.fixture
def storage(tmp_path):
    with open_storage(tmp_path / "storage", mode="w") as storage:
        yield storage

import numpy.typing as npt
import pandas as pd

rng = np.random.default_rng()

Coordinate = namedtuple("Coordinate", ["x", "y"])


def make_observations(
        coordinates: List[Coordinate],
        times: npt.NDArray[np.int_],
        field: npt.NDArray[np.float_],
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
            _df = pd.DataFrame(
                {
                    "k": [k],
                    "x": [coordinate.x],
                    "y": [coordinate.y],
                    "value": [value + rng.normal(loc=0.0, scale=sd)],
                    "sd": [sd],
                }
            )
            d = pd.concat([d, _df])
    d = d.set_index(["k", "x", "y"], verify_integrity=True)

    return d


import numpy as np
import scipy.linalg as sla


def gaussian_fields(pts, rng, N=1, r=0.2):
    """Random field generation.

    Uses:
    - Gaussian variogram.
    - Gaussian distributions.
    """
    def dist_euclid(X):
        """Compute distances.

        X must be a 2D-array of shape `(nPt, nDim)`.

        Note: not periodic.
        """
        diff = X[:, None, :] - X
        d2 = np.sum(diff**2, axis=-1)
        return np.sqrt(d2)

    def vectorize(*XYZ):
        """Reshape coordinate points.

        Input: `nDim` arrays with equal `shape`.
        Let `nPt = np.prod(shape)`
        Output: array of shape `(nPt, nDim)`.
        """
        return np.stack(XYZ).reshape((len(XYZ), -1)).T

    def variogram_gauss(xx, r, n=0, a=1 / 3):
        """Compute the Gaussian variogram for the 1D points xx.

        Params:
        range r, nugget n, and a

        Ref:
        https://en.wikipedia.org/wiki/Variogram#Variogram_models

        Example:
        >>> xx = np.array([0, 1, 2])
        >>> variogram_gauss(xx, 1, n=0.1, a=1)
        array([0.        , 0.6689085 , 0.98351593])
        """
        # Gauss
        gamma = 1 - np.exp(-(xx**2) / r**2 / a)
        # Sill (=1)
        gamma *= 1 - n
        # Nugget
        gamma[xx != 0] += n
        return gamma

    dists = dist_euclid(vectorize(*pts))
    Cov = 1 - variogram_gauss(dists, r)
    C12 = sla.sqrtm(Cov).real
    fields = rng.standard_normal(size=(N, len(dists))) @ C12.T
    return fields


def heat_equation(
        u,
        alpha,
        dx: int,
        dt: float,
        k_start: int,
        k_end: int,
        rng: np.random.Generator,
        scale = None,
):
    """2D heat equation that supports field of heat coefficients.

    Based on:
    https://levelup.gitconnected.com/solving-2d-heat-equation-numerically-using-python-3334004aa01a
    """
    _u = u.copy()
    # assert (dt <= dx**2 / (4 * alpha)).all(), "Choise of dt not numerically stable"
    nx = u.shape[1]  # number of grid cells
    assert alpha.shape == (nx, nx)

    gamma = (alpha * dt) / (dx**2)
    plate_length = u.shape[1]
    for k in range(k_start, k_end - 1, 1):
        for i in range(1, plate_length - 1, dx):
            for j in range(1, plate_length - 1, dx):
                if scale is not None:
                    noise = rng.normal(scale=scale)
                else:
                    noise = 0
                _u[k + 1, i, j] = (
                        gamma[i, j]
                        * (
                                _u[k][i + 1][j]
                                + _u[k][i - 1][j]
                                + _u[k][i][j + 1]
                                + _u[k][i][j - 1]
                                - 4 * _u[k][i][j]
                        )
                        + _u[k][i][j]
                        + noise
                )

    return _u


def sample_prior_perm(N, nx):
    mesh = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, nx))
    lperms = np.exp(gaussian_fields(mesh, rng, N, r=0.8))
    return lperms

@pytest.mark.integration_test
def test_surface_update_heat_equation(tmpdir):
    """
    This does an update using the heat equation
    """
    # Define true parameters, set true initial conditions and calculate the true temperature field
    # Perhaps obvious, but we do not have this information in real-life.
    # Evensens' formulation of the Ensemble Smoother has the prior as
    # a (nx * nx, N) matrix, i.e (number of parameters, N).
    dx = 1
    nx = 10
    # time steps
    k_start = 0
    k_end = 500

    # Set the coefficient of heat transfer for each grid cell.
    alpha_t = sample_prior_perm(1, nx).T.reshape(nx, nx)

    # Calculate maximum `dt`.
    # If higher values are used, the numerical solution will become unstable.
    dt = dx**2 / (4 * np.max(alpha_t))

    # True initial temperature field.
    u_init = np.empty((k_end, nx, nx))
    u_init.fill(0.0)

    # Heating the plate at two points initially.
    # How you define initial conditions will effect the spread of results,
    # i.e., how similar different realisations are.
    u_init[:, 5, 5] = 100
    # u_init[:, 7, 7] = 100

    # How much noise to add to heat equation, also called model noise.
    scale = None
    num_realizations = 50
    u_t = heat_equation(u_init, alpha_t, dx, dt, k_start, k_end, rng=rng, scale=scale)
    obs_coordinates = [
        Coordinate(5, 3),
        Coordinate(3, 5),
        Coordinate(5, 7),
        Coordinate(7, 5),
        Coordinate(2, 2),
        Coordinate(7, 2),
    ]
    k_end = 500
    obs_times = np.linspace(5, k_end, 8, endpoint=False, dtype=int)
    d = make_observations(obs_coordinates, obs_times, u_t, lambda value: abs(0.05 * value))
    obs_times = sorted(set(d.index.get_level_values("k").to_list()))
    report_steps = ",".join(map(str, obs_times))
    index = sorted((obs.x, obs.y) for obs in obs_coordinates)
    with tmpdir.as_cwd():
        config = dedent(
            f"""
            NUM_REALIZATIONS {num_realizations}
            OBS_CONFIG observations
            SURFACE MY_PARAM OUTPUT_FILE:surf.irap INIT_FILES:surf.irap BASE_SURFACE:base_surf.irap FORWARD_INIT:True
            GEN_DATA MY_RESPONSE RESULT_FILE:gen_data_%d.out REPORT_STEPS:{report_steps} INPUT_FORMAT:ASCII
            INSTALL_JOB heat_equation HEAT_EQUATION
            SIMULATION_JOB heat_equation
            TIME_MAP time_map
            QUEUE_OPTION LOCAL MAX_RUNNING 10
        """  # pylint: disable=line-too-long  # noqa: E501
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)

        with open("forward_model", "w", encoding="utf-8") as f:
            f.write(
                dedent(
                    f"""#!/usr/bin/env python
import xtgeo
import numpy as np
import os
import scipy.linalg as sla

rng = np.random.default_rng()

# Number of grid-cells in x and y direction
nx = {nx}
dx = {dx}

# time steps
k_start = {k_start}
k_end = {k_end}

{inspect.getsource(heat_equation)}

{inspect.getsource(gaussian_fields)}

{inspect.getsource(sample_prior_perm)}

if __name__ == "__main__":
    if not os.path.exists("surf.irap"):
        # Evensens' formulation of the Ensemble Smoother has the prior as
        # a (nx * nx, N) matrix, i.e (number of parameters, N).
        prior = sample_prior_perm(1, nx).T.flatten()

        surf = xtgeo.RegularSurface(ncol=nx,
                                    nrow=nx,
                                    xinc=1,
                                    yinc=1,
                                    values=prior)
        surf.to_file("surf.irap", fformat="irap_ascii")
    parameter = xtgeo.surface_from_file("surf.irap", fformat="irap_ascii")
    alpha = parameter.values
    # True initial temperature field.
    u_init = np.empty((k_end, nx, nx))
    u_init.fill(0.0)
    # Heating the plate at two points initially.
    # How you define initial conditions will effect the spread of results,
    # i.e., how similar different realisations are.
    u_init[:, 5, 5] = 100
    # Calculate maximum `dt`.
    # If higher values are used, the numerical solution will become unstable.
    dt = dx**2 / (4 * np.max(alpha))

    response = heat_equation(u_init, alpha, dx, dt, k_start, k_end, rng=rng, scale=None)
    index = {index}
    for time_step in {obs_times}:
        with open(f"gen_data_{{time_step}}.out", "w", encoding="utf-8") as f:
            f.write("\\n".join(str(response[time_step][i]) for i in index))
        """
                )
            )
        os.chmod(
            "forward_model",
            os.stat("forward_model").st_mode
            | stat.S_IXUSR
            | stat.S_IXGRP
            | stat.S_IXOTH,
        )
        with open("HEAT_EQUATION", "w", encoding="utf-8") as fout:
            fout.write("EXECUTABLE forward_model")
        with open("observations", "w", encoding="utf-8") as fout:
            for obs_time in obs_times:
                fout.write(
                    dedent(
                        f"""
                GENERAL_OBSERVATION MY_OBS_{obs_time} {{
                    DATA       = MY_RESPONSE;
                    RESTART    = {obs_time};
                    OBS_FILE   = obs_{obs_time}.txt;
                }};"""
                    )
                )
        for obs_time in obs_times:
            with open(f"obs_{obs_time}.txt", "w", encoding="utf-8") as fobs:
                df = d.iloc[d.index.get_level_values('k') == obs_time]
                fobs.write(df.sort_index().to_csv(header=False, index=False, sep=" "))

        with open("time_map", "w", encoding="utf-8") as fobs:
            for obs_time in obs_times:
                fobs.write((datetime(2014, 9, 14) + timedelta(days=obs_time)).strftime("%Y-%m-%d") + "\n")

        base_surface = xtgeo.RegularSurface(
            ncol=nx,
            nrow=nx,
            xinc=1,
            yinc=1,
            values=alpha_t.flatten()
        )
        base_surface.to_file("base_surf.irap", fformat="irap_ascii")

        parser = ArgumentParser(prog="test_main")
        parsed = ert_parser(
            parser,
            [
                ENSEMBLE_SMOOTHER_MODE,
                "--current-case",
                "prior",
                "--target-case",
                "smoother_update",
                "config.ert",
                "--port-range",
                "1024-65535",
            ],
        )

        run_cli(parsed)
        ert = EnKFMain(ErtConfig.from_file("config.ert"))
        priors = []
        posteriors = []
        for i in range(50):
            prior = xtgeo.surface_from_file(f"simulations/realization-{i}/iter-0/surf.irap", fformat="irap_ascii")
            posterior = xtgeo.surface_from_file(f"simulations/realization-{i}/iter-1/surf.irap", fformat="irap_ascii")
            prior.quickplot(f"prior_{i}")
            posterior.quickplot(f"posterior_{i}")
            priors.append(prior.values)
            posteriors.append(posterior.values)
        prior_mean = np.mean(priors, axis=0)
        posterior_mean = np.mean(posteriors, axis=0)

        posterior.set_values1d(posterior_mean.flatten())
        prior.set_values1d(prior_mean.flatten())

        posterior.quickplot("posterior_mean", colormap="viridis")
        prior.quickplot("prior_mean", colormap="viridis")

        base_surface.quickplot("exact", colormap="viridis")
