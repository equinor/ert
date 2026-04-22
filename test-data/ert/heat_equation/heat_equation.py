#!/usr/bin/env python3
"""Partial Differential Equations to use as forward models."""

import datetime
import json
import sys
from pathlib import Path

import geostat
import numpy as np
import numpy.typing as npt
import resfo
from definition import (
    dx,
    k_end,
    k_start,
    nx,
    nz,
    obs_coordinates,
    obs_times,
    summary_names,
    u_init,
)
from resfo_utilities.testing import (
    Date,
    Simulator,
    Smspec,
    SmspecIntehead,
    SummaryMiniStep,
    SummaryStep,
    UnitSystem,
    Unsmry,
)


def create_summary_smspec_unsmry(
    summary_vectors: dict[str, list[float]],
    start_date: datetime.date,
    time_step_in_days: float = 30,
):
    summary_keys = list(summary_vectors.keys())
    num_time_steps = len(summary_vectors[summary_keys[0]])

    unsmry = Unsmry(
        steps=[
            SummaryStep(
                seqnum=0,
                ministeps=[
                    SummaryMiniStep(
                        mini_step=0,
                        params=[
                            24 * time_step_in_days * step,
                            *[summary_vectors[key][step] for key in summary_vectors],
                        ],
                    )
                ],
            )
            for step in range(num_time_steps)
        ]
    )
    smspec = Smspec(
        nx=nx,
        ny=nx,
        nz=nz,
        restarted_from_step=0,
        num_keywords=1 + len(summary_keys),
        restart="        ",
        keywords=["TIME    ", *summary_keys],
        well_names=[":+:+:+:+", *([":+:+:+:+"] * len(summary_keys))],
        region_numbers=[-32676, *([0] * len(summary_keys))],
        units=["HOURS   ", *(["SM3"] * len(summary_keys))],
        start_date=Date.from_datetime(start_date),
        intehead=SmspecIntehead(
            unit=UnitSystem.METRIC,
            simulator=Simulator.ECLIPSE_100,
        ),
    )

    return smspec, unsmry


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
    """3D heat equation that supports a field of heat coefficients.

    Solves the heat equation on a (nx, nx, nz) grid using explicit
    finite differences.  When nz == 1 the z-stencil terms vanish and
    the solver reduces to the classic 2D formulation.

    Based on:
    https://levelup.gitconnected.com/solving-2d-heat-equation-numerically-using-python-3334004aa01a
    """
    u_ = u.copy()
    nx = u.shape[1]  # number of grid cells in x and y
    n_z = u.shape[3]  # number of grid cells in z
    assert cond.shape == (nx, nx, n_z)
    gamma = (cond * dt) / (dx**2)

    # Pre-sample all noise at once
    if scale is not None:
        num_steps = k_end - k_start - 1
        noise_all = rng.normal(0, scale, size=(num_steps, nx - 2, nx - 2, n_z))

    for k in range(k_start, k_end - 1):
        # Vectorized finite difference in x and y (interior cells)
        xy_stencil = (
            gamma[1:-1, 1:-1, :]
            * (
                u_[k, 2:, 1:-1, :]  # i+1
                + u_[k, :-2, 1:-1, :]  # i-1
                + u_[k, 1:-1, 2:, :]  # j+1
                + u_[k, 1:-1, :-2, :]  # j-1
                - 4 * u_[k, 1:-1, 1:-1, :]
            )
            + u_[k, 1:-1, 1:-1, :]
        )

        if n_z > 1:
            # z-direction stencil for interior z-layers
            z_contrib = np.zeros_like(xy_stencil)
            z_contrib[:, :, 1:-1] = (
                gamma[1:-1, 1:-1, 1:-1]
                * (
                    u_[k, 1:-1, 1:-1, 2:]  # l+1
                    + u_[k, 1:-1, 1:-1, :-2]  # l-1
                    - 2 * u_[k, 1:-1, 1:-1, 1:-1]
                )
            )
            # z-boundary layers: zero Dirichlet (u=0 at ghost cell)
            z_contrib[:, :, 0] = gamma[1:-1, 1:-1, 0] * (
                u_[k, 1:-1, 1:-1, 1] - 2 * u_[k, 1:-1, 1:-1, 0]
            )
            z_contrib[:, :, -1] = gamma[1:-1, 1:-1, -1] * (
                u_[k, 1:-1, 1:-1, -2] - 2 * u_[k, 1:-1, 1:-1, -1]
            )
            xy_stencil += z_contrib

        u_[k + 1, 1:-1, 1:-1, :] = xy_stencil

        # Add pre-sampled noise
        if scale is not None:
            noise_idx = k - k_start
            u_[k + 1, 1:-1, 1:-1, :] += noise_all[noise_idx]

    return u_


def sample_prior_conductivity(ensemble_size, nx, nz, rng, corr_length):
    mesh = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, nx))
    # Generate an independent 2D conductivity field for each z-layer so
    # that the prior has genuine z-variation. This is necessary for the
    # EnKF to produce layer-differentiated posteriors and for the
    # localization z-ordering to matter.
    layers = [
        np.exp(geostat.gaussian_fields(mesh, rng, ensemble_size, r=corr_length))
        for _ in range(nz)
    ]
    return np.stack(layers, axis=-1)


def load_parameters(filename):
    with Path(filename).open(encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    iens = int(sys.argv[1])
    iteration = int(sys.argv[2])
    rng = np.random.default_rng(iens)

    parameters = load_parameters("parameters.json")

    if iteration == 0:
        cond = sample_prior_conductivity(
            ensemble_size=1,
            nx=nx,
            nz=nz,
            rng=rng,
            corr_length=float(parameters["x"]["value"]),
        ).reshape(nx, nx, nz)

        resfo.write(
            "cond.bgrdecl", [("COND    ", cond.flatten(order="F").astype(np.float32))]
        )
    else:
        cond = resfo.read("cond.bgrdecl")[0][1].reshape(nx, nx, nz, order="F")

    # The update may give non-physical parameter values, which here means
    # negative heat conductivity. Setting negative values to a small positive
    # value but not zero because we want to be able to divide by them.
    cond = cond.clip(min=1e-8)

    # Calculate maximum `dt`.
    # If higher values are used, the numerical solution will become unstable.
    # Note that this could be avoided if we used an implicit solver.
    # The stability bound for a 3D explicit scheme is dx^2 / (6 * max(cond)),
    # but when nz == 1 the z-stencil vanishes so dx^2 / (4 * max(cond)) suffices.
    neighbors = 6 if nz > 1 else 4
    dt = dx**2 / (neighbors * np.max(cond))

    scaled_u_init = u_init * float(parameters["t"]["value"])

    response = heat_equation(scaled_u_init, cond, dx, dt, k_start, k_end, rng)

    # Observations are extracted per z-layer
    index = sorted((obs.x, obs.y) for obs in obs_coordinates)
    for time_step in obs_times:
        with Path(f"gen_data_{time_step}.out").open("w", encoding="utf-8") as f:
            for i in index:
                f.writelines(f"{response[time_step][i][z]}\n" for z in range(nz))

    time_map = []
    start_date = datetime.date(2010, 1, 1)

    summary_values = {name: [] for name in summary_names}
    for time_step in range(k_start, k_end):
        time_map.append(
            (start_date + datetime.timedelta(days=float(time_step))).isoformat()
        )
        for obs in obs_coordinates:
            for z in range(nz):
                summary_values[f"HEAT_{obs.x}_{obs.y}_{z}"].append(
                    response[time_step][obs.x, obs.y, z]
                )

    smspec, unsmry = create_summary_smspec_unsmry(
        summary_vectors=summary_values,
        start_date=datetime.datetime(2010, 1, 1),  # noqa: DTZ001
        time_step_in_days=1,
    )

    smspec.to_file("HEAT.SMSPEC")
    unsmry.to_file("HEAT.UNSMRY")
