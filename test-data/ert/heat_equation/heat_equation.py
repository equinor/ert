#!/usr/bin/env python3
"""Partial Differential Equations to use as forward models."""

import datetime
import json
import sys

import geostat
import numpy as np
import numpy.typing as npt
import resfo
from definition import (
    dx,
    k_end,
    k_start,
    nx,
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
        nz=1,
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
    """2D heat equation that suppoheat_erts field of heat coefficients.

    Based on:
    https://levelup.gitconnected.com/solving-2d-heat-equation-numerically-using-python-3334004aa01a
    """
    u_ = u.copy()
    nx = u.shape[1]  # number of grid cells
    assert cond.shape == (nx, nx)
    gamma = (cond * dt) / (dx**2)

    # Pre-sample all noise at once
    if scale is not None:
        num_steps = k_end - k_start - 1
        noise_all = rng.normal(0, scale, size=(num_steps, nx - 2, nx - 2))

    for k in range(k_start, k_end - 1):
        # Vectorized finite difference
        u_[k + 1, 1:-1, 1:-1] = (
            gamma[1:-1, 1:-1]
            * (
                u_[k, 2:, 1:-1]  # i+1
                + u_[k, :-2, 1:-1]  # i-1
                + u_[k, 1:-1, 2:]  # j+1
                + u_[k, 1:-1, :-2]  # j-1
                - 4 * u_[k, 1:-1, 1:-1]
            )
            + u_[k, 1:-1, 1:-1]
        )
        # Add pre-sampled noise
        if scale is not None:
            noise_idx = k - k_start
            u_[k + 1, 1:-1, 1:-1] += noise_all[noise_idx]

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

    if iteration == 0:
        cond = sample_prior_conductivity(
            ensemble_size=1, nx=nx, rng=rng, corr_length=float(parameters["x"]["value"])
        ).reshape(nx, nx)

        resfo.write(
            "cond.bgrdecl", [("COND    ", cond.flatten(order="F").astype(np.float32))]
        )
    else:
        cond = resfo.read("cond.bgrdecl")[0][1].reshape(nx, nx)

    # The update may give non-physical parameter values, which here means
    # negative heat conductivity. Setting negative values to a small positive
    # value but not zero because we want to be able to divide by them.
    cond = cond.clip(min=1e-8)

    # Calculate maximum `dt`.
    # If higher values are used, the numerical solution will become unstable.
    # Note that this could be avoided if we used an implicit solver.
    dt = dx**2 / (4 * np.max(cond))

    scaled_u_init = u_init * float(parameters["t"]["value"])

    response = heat_equation(scaled_u_init, cond, dx, dt, k_start, k_end, rng)

    index = sorted((obs.x, obs.y) for obs in obs_coordinates)
    for time_step in obs_times:
        with open(f"gen_data_{time_step}.out", "w", encoding="utf-8") as f:
            f.writelines(f"{response[time_step][i]}\n" for i in index)

    time_map = []
    start_date = datetime.date(2010, 1, 1)

    summary_values = {name: [] for name in summary_names}
    # for time_step in obs_times:
    for time_step in range(k_start, k_end):
        time_map.append(
            (start_date + datetime.timedelta(days=float(time_step))).isoformat()
        )
        for obs in obs_coordinates:
            summary_values[f"HEAT_{obs.x}_{obs.y}"].append(
                response[time_step][obs.x, obs.y]
            )

    smspec, unsmry = create_summary_smspec_unsmry(
        summary_vectors=summary_values,
        start_date=datetime.datetime(2010, 1, 1),
        time_step_in_days=1,
    )

    smspec.to_file("HEAT.SMSPEC")
    unsmry.to_file("HEAT.UNSMRY")
