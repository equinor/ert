#!/usr/bin/env python3
"""Partial Differential Equations to use as forward models."""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import gaussianfft as grf
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
    room_temperature,
    summary_names,
    u_init,
)


@dataclass(frozen=True)
class HeatEquationOutputs:
    summary_values: dict[str, list[float]]
    response_layers: dict[int, npt.NDArray[np.float64]]


def write_unsmry(
    summary_vectors: dict[str, list[float]],
    output_file: str | Path = "HEAT.UNSMRY",
    time_step_in_days: float = 30,
) -> None:
    summary_keys = list(summary_vectors.keys())
    num_time_steps = len(summary_vectors[summary_keys[0]])

    unsmry_records: list[
        tuple[str, npt.NDArray[np.int32] | npt.NDArray[np.float32]]
    ] = []
    for step in range(num_time_steps):
        unsmry_records.extend(
            [
                ("SEQHDR  ", np.array([0], dtype=np.int32)),
                ("MINISTEP", np.array([0], dtype=np.int32)),
                (
                    "PARAMS  ",
                    np.array(
                        [
                            24 * time_step_in_days * step,
                            *[summary_vectors[key][step] for key in summary_keys],
                        ],
                        dtype=np.float32,
                    ),
                ),
            ]
        )

    resfo.write(output_file, unsmry_records)


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


def simulate_heat_equation_outputs(
    initial_temperature: npt.NDArray[np.float64],
    cond: npt.NDArray[np.float64],
    dx: int,
    dt: float,
    k_start: int,
    k_end: int,
) -> HeatEquationOutputs:
    """Run the solver while keeping only the data written by this forward model."""
    current = initial_temperature.copy()
    next_layer = current.copy()
    nx = current.shape[0]
    assert cond.shape == (nx, nx)
    gamma = (cond * dt) / (dx**2)

    summary_values = {
        name: [current[obs.x, obs.y]]
        for name, obs in zip(summary_names, obs_coordinates, strict=True)
    }
    response_layers: dict[int, npt.NDArray[np.float64]] = {}
    obs_time_steps = {int(time_step) for time_step in obs_times}
    if k_start in obs_time_steps:
        response_layers[k_start] = current.copy()

    for k in range(k_start, k_end - 1):
        next_layer[1:-1, 1:-1] = (
            gamma[1:-1, 1:-1]
            * (
                current[2:, 1:-1]
                + current[:-2, 1:-1]
                + current[1:-1, 2:]
                + current[1:-1, :-2]
                - 4 * current[1:-1, 1:-1]
            )
            + current[1:-1, 1:-1]
        )
        current, next_layer = next_layer, current

        time_step = k + 1
        for name, obs in zip(summary_names, obs_coordinates, strict=True):
            summary_values[name].append(current[obs.x, obs.y])
        if time_step in obs_time_steps:
            response_layers[time_step] = current.copy()

    return HeatEquationOutputs(
        summary_values=summary_values,
        response_layers=response_layers,
    )


def sample_prior_conductivity(ensemble_size, nx, rng, corr_length):
    dx = 1.0 / (nx - 1)
    grf.seed(int(rng.integers(0, 2**31)))
    variogram = grf.variogram(grf.VariogramType.EXPONENTIAL, corr_length)
    fields = np.stack(
        [
            np.reshape(
                grf.simulate(variogram, nx=nx, dx=dx, ny=nx, dy=dx), (nx, nx), order="F"
            )
            for _ in range(ensemble_size)
        ]
    )
    return np.exp(fields)


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
            ensemble_size=1, nx=nx, rng=rng, corr_length=float(parameters["x"]["value"])
        ).reshape(nx, nx)

        resfo.write(
            "cond.bgrdecl", [("COND    ", cond.flatten(order="F").astype(np.float32))]
        )
    else:
        cond = resfo.read("cond.bgrdecl")[0][1].reshape(nx, nx, order="F")

    # The update may give non-physical parameter values, which here means
    # negative heat conductivity. Setting negative values to a small positive
    # value but not zero because we want to be able to divide by them.
    cond = cond.clip(min=1e-8)

    # Calculate maximum `dt`.
    # If higher values are used, the numerical solution will become unstable.
    # Note that this could be avoided if we used an implicit solver.
    dt = dx**2 / (4 * np.max(cond))

    t = float(parameters["t"]["value"])
    initial_temperature = room_temperature + (u_init[0] - room_temperature) * t
    outputs = simulate_heat_equation_outputs(
        initial_temperature,
        cond,
        dx,
        dt,
        k_start,
        k_end,
    )

    index = sorted((obs.x, obs.y) for obs in obs_coordinates)
    for time_step in obs_times:
        with Path(f"gen_data_{time_step}.out").open("w", encoding="utf-8") as f:
            f.writelines(
                f"{outputs.response_layers[int(time_step)][i]}\n" for i in index
            )

    write_unsmry(outputs.summary_values, time_step_in_days=1)
