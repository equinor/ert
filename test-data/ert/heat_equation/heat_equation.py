#!/usr/bin/env python3

import json
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import resfo

from iterative_ensemble_smoother._experimental import sample_prior

def heat_equation(
    u_init: npt.NDArray[np.float64],
    conductivity: npt.NDArray[np.float64],
    dx: float,
    dt: float,
    k_start: int,
    k_end: int,
    rng: np.random.Generator,
    scale: float | None = None,
) -> npt.NDArray[np.float64]:
    """
    Solves the 2D heat equation using finite differences.
    """
    nx = u.shape[1] # number of grid cells
    u_ = np.zeros((k_end, nx, nx))
    u_[k_start] = u_init

    gamma = conductivity * dt / (dx**2)

    # Pre-sample all noise at once
    if scale is not None:
        num_steps = k_end - k_start - 1
        noise_all = rng.normal(0, scale, size=(num_steps, nx - 2, nx - 2))

    for k in range(k_start, k_end - 1):
        # Vectorized finite difference
        u_[k + 1, 1:-1, 1:-1] = (
            gamma[1:-1, 1:-1]
            * (
                u_[k, 2:, 1:-1] # i+1
                + u_[k, :-2, 1:-1] # i-1
                + u_[k, 1:-1, 2:] # j+1
                + u_[k, 1:-1, :-2] # j-1
                - 4 * u_[k, 1:-1, 1:-1]
            )
            + u_[k, 1:-1, 1:-1]
        )
        # Add pre-sampled noise
        if scale is not None:
            noise_idx = k - k_start
            u_[k + 1, 1:-1, 1:-1] += noise_all[noise_idx]

    return u_


mesh = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, nx))


def load_parameters(filename):
    with open(filename, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("Usage: heat_equation.py nx realization")

    nx = int(sys.argv[1])
    realization = int(sys.argv[2])
    iteration = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    parameters = load_parameters("parameters.json")
    init_temp_scale = parameters["INIT_TEMP_SCALE"]
    corr_length = parameters["CORR_LENGTH"]

    if iteration == 0:
        # Generate conductivity field with improved error handling
        rng = np.random.default_rng(seed=1234 + realization)
        try:
            cond_flat = sample_prior(
                ensemble_size=1, nx=nx, rng=rng, corr_length=float(corr_length["x"])
            )
            
            # Improved error handling for reshape operation
            expected_size = nx * nx
            if cond_flat.size != expected_size:
                raise ValueError(
                    f"Surface size mismatch: Generated conductivity field has {cond_flat.size} elements, "
                    f"but expected {expected_size} elements for a {nx}x{nx} grid. "
                    f"This may be caused by inconsistent surface dimensions between base surface and generated surface."
                )
            
            cond = cond_flat.reshape(nx, nx)
            
        except ValueError as e:
            if "cannot reshape array" in str(e):
                # Convert the generic reshape error to a more informative one
                raise ValueError(
                    f"Surface size mismatch: Cannot reshape conductivity array of size {cond_flat.size} "
                    f"into shape ({nx}, {nx}). Expected {nx*nx} elements but got {cond_flat.size}. "
                    f"This indicates a mismatch between the base surface dimensions and the generated surface."
                ) from e
            else:
                # Re-raise other ValueError types with additional context
                raise ValueError(f"Error processing conductivity surface: {e}") from e

        resfo.write(
            "conductivity.roff",
            grid=(nx, nx, 1),
            props={"CONDUCTIVITY": cond.ravel()},
        )

        # negative heat conductivity. Setting negative values to a small
        # value but not zero because we want to be
        cond = cond.clip(min=1e-8)

        rng = np.random.default_rng(seed=1234 + realization)
        scaled_u_init = init_temp_scale * rng.uniform(0, 1, size=(nx, nx))
    else:
        # Load conductivity from file with error handling
        try:
            props_cond = resfo.read("conductivity.roff")
            cond_flat = props_cond["CONDUCTIVITY"]
            
            # Check size before reshaping
            expected_size = nx * nx
            if len(cond_flat) != expected_size:
                raise ValueError(
                    f"Loaded conductivity data size mismatch: Found {len(cond_flat)} elements, "
                    f"but expected {expected_size} elements for a {nx}x{nx} grid."
                )
            
            cond = cond_flat.reshape((nx, nx))
            
        except Exception as e:
            raise RuntimeError(f"Failed to load conductivity surface from file: {e}") from e

        cond = cond.clip(min=1e-8)
        scaled_u_init = init_temp_scale * rng.uniform(0, 1, size=(nx, nx))

    response = heat_equation(scaled_u_init, cond, dx, dt, k_start, k_end, rng)

    # Write response with error handling
    try:
        resfo.write(
            "temperature.roff",
            grid=(nx, nx, 1),
            props={"TEMPERATURE": response[-1].ravel()},
        )
    except Exception as e:
        raise RuntimeError(f"Failed to write temperature surface to file: {e}") from e
