from typing import NamedTuple

import numpy as np

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

# Resolution in the x-direction (nothing to worry about really)
dx = 1


class Coordinate(NamedTuple):
    x: int
    y: int


obs_coordinates = [
    Coordinate(5, 3),
    Coordinate(3, 5),
    Coordinate(5, 7),
    Coordinate(7, 5),
    Coordinate(2, 2),
    Coordinate(7, 2),
]

summary_names = [f"HEAT_{coord.x}_{coord.y}" for coord in obs_coordinates]

obs_times = np.linspace(10, k_end, 8, endpoint=False, dtype=int)
