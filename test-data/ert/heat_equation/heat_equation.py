#!/usr/bin/env python3
"""Partial Differential Equations to use as forward models."""

import json
import sys

import geostat
import numpy as np
import numpy.typing as npt
import resfo
from definition import dx, k_end, k_start, nx, obs_coordinates, obs_times, u_init


def heat_equation(
    u: npt.NDArray[np.float64],
    cond: npt.NDArray[np.float64],
    dx: int,
    dt: float,
    k_start: int,
    k_end: int,
    rng: np.random.Generator,
    scale: float | None = None,
) -> npt.NDArray[np.float64]:
    """2D heat equation that suppoheat_erts field of heat coefficients.

    Based on:
    https://levelup.gitconnected.com/solving-2d-heat-equation-numerically-using-python-3334004aa01a
    """
    u_ = u.copy()
    nx = u.shape[1]  # number of grid cells
    assert cond.shape == (nx, nx)

    gamma = (cond * dt) / (dx**2)
    plate_length = u.shape[1]
    for k in range(k_start, k_end - 1, 1):
        for i in range(1, plate_length - 1, dx):
            for j in range(1, plate_length - 1, dx):
                noise = rng.normal(scale=scale) if scale is not None else 0
                u_[k + 1, i, j] = (
                    gamma[i, j]
                    * (
                        u_[k][i + 1][j]
                        + u_[k][i - 1][j]
                        + u_[k][i][j + 1]
                        + u_[k][i][j - 1]
                        - 4 * u_[k][i][j]
                    )
                    + u_[k][i][j]
                    + noise
                )

    return u_


def sample_prior_conductivity(ensemble_size, nx, rng, corr_length):
    mesh = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, nx))
    return np.exp(geostat.gaussian_fields(mesh, rng, ensemble_size, r=corr_length))


def load_parameters(filename):
    with open(filename, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    iens = int(sys.argv[1])
    iteration = int(sys.argv[2])
    rng = np.random.default_rng(iens)

    parameters = load_parameters("parameters.json")
    init_temp_scale = parameters["INIT_TEMP_SCALE"]
    corr_length = parameters["CORR_LENGTH"]

    cond = sample_prior_conductivity(
        ensemble_size=1, nx=nx, rng=rng, corr_length=float(corr_length["x"])
    ).reshape(nx, nx)

    if iteration == 0:
        resfo.write(
            "cond.bgrdecl", [("COND    ", cond.flatten(order="F").astype(np.float32))]
        )
    else:
        cond = resfo.read("cond.bgrdecl")[0][1].reshape(nx, nx)

    # The update may give non-physical parameter values, which here means negative heat conductivity.
    # Setting negative values to a small positive value but not zero because we want to be able to divide by them.
    cond = cond.clip(min=1e-8)

    # Calculate maximum `dt`.
    # If higher values are used, the numerical solution will become unstable.
    # Note that this could be avoided if we used an implicit solver.
    dt = dx**2 / (4 * max(np.max(cond), np.max(cond)))

    scaled_u_init = u_init * float(init_temp_scale["x"])

    response = heat_equation(scaled_u_init, cond, dx, dt, k_start, k_end, rng)

    index = sorted((obs.x, obs.y) for obs in obs_coordinates)
    for time_step in obs_times:
        with open(f"gen_data_{time_step}.out", "w", encoding="utf-8") as f:
            for i in index:
                f.write(f"{response[time_step][i]}\n")
