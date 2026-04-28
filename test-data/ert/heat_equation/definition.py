from typing import NamedTuple

import numpy as np

# Number of grid-cells in x and y direction
nx = 50

# time steps
k_start = 0
k_end = 500

# Define initial condition, i.e., the initial temperature distribution.
# How you define initial conditions will effect the spread of results,
# i.e., how similar different realisations are.
room_temperature = -25.0
u_init = np.full((k_end, nx, nx), room_temperature)
u_init[:, 13:38, 13:38] = 100

# Resolution in the x-direction (nothing to worry about really)
dx = 1


class Coordinate(NamedTuple):
    x: int
    y: int


obs_coordinates = [
    Coordinate(10, 25),
    Coordinate(25, 10),
    Coordinate(40, 25),
    Coordinate(25, 40),
]

summary_names = [f"HEAT_{coord.x}_{coord.y}" for coord in obs_coordinates]

obs_times = np.linspace(10, k_end, 8, endpoint=False, dtype=int)
