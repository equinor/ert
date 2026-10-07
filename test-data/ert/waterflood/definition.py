"""Definition of the waterflood case, shared by the forward model and the generator.

The reservoir is a 1000 x 1000 x 20 m box, discretized areally into 50 x 50 cells,
in metric units (m, day, bar, mD, cP). One water injector floods four producers held on
bottom-hole pressure, in a five-spot pattern. Everything is deterministic except the
uncertain parameters: the permeability field, the range of its variogram, and the
residual oil saturation.
"""

from typing import NamedTuple

import numpy as np

# Grid: areal discretization of a box of thickness `dz`
nx = ny = 50
dx = 20.0  # cell size [m]
dz = 20.0  # reservoir thickness [m]
lx = nx * dx  # extent [m]

# Rock
poro = 0.2  # porosity [-]
ct = 1e-5  # total (rock + fluid) compressibility [1/bar]

# The permeability field is log-normal, with a Gaussian variogram whose range is
# the uncertain CORR_LENGTH [m]. The mean and the (log) standard deviation are fixed.
perm_mean = 200.0  # [mD]
perm_std = 0.8  # standard deviation of log(K)

# Fluids: water flooding a five times more viscous oil, quadratic Corey curves.
# The residual oil saturation is the uncertain SOR.
visc_water = 1.0  # [cP]
visc_oil = 5.0  # [cP]
swc = 0.2  # connate water saturation [-]

# Wells: rate on the injector, bottom-hole pressure on the producers
rate_inj = 1000.0  # total injection [m3/day]
bhp_prod = 250.0  # [bar]
pressure_init = 300.0  # [bar]
well_radius = 0.15  # [m]


class Well(NamedTuple):
    name: str
    east: float
    north: float


injectors = [Well("INJ", 0.5 * lx, 0.5 * lx)]
producers = [
    Well("PROD1", 0.15 * lx, 0.15 * lx),
    Well("PROD2", 0.85 * lx, 0.15 * lx),
    Well("PROD3", 0.15 * lx, 0.85 * lx),
    Well("PROD4", 0.85 * lx, 0.85 * lx),
]
# An ordinary five-spot: the injector at the centre, a producer towards each corner.
# Four breakthrough times are few for 2500 cells, so the update needs localization to
# make anything of them -- which is half of what this case is for (ref the README).

# Schedule: 40 report steps of 60 days, i.e. 2400 days, over which about 0.6 pore
# volumes are injected, so that every producer sees water before the end.
dt = 60.0  # [days]
nsteps = 40
start_date = "2020-01-01"

# Responses: ECLIPSE-style summary vectors. The water cut and the oil rate of each
# producer, and the bottom-hole pressure each injector needs to sustain its rate.
summary_keys = (
    [f"WWCT:{well.name}" for well in producers]
    + [f"WOPR:{well.name}" for well in producers]
    + [f"WBHP:{well.name}" for well in injectors]
)

# History and forecast: the case is observed over its first `nsteps_history` steps
# only, and simulated to the end, so that what the update did to the *prediction*
# can be scored on the steps that were not assimilated. In the truth, PROD1 breaks
# history match is run to answer: when does the last producer water out?
nsteps_history = 16
obs_steps = np.arange(4, nsteps_history + 1, 4)  # every fourth step of the history
forecast_steps = np.arange(nsteps_history + 1, nsteps + 1)

# The truth: a realization of the prior, at the mean correlation length, which
# `generate_files.py` perturbs into the observations and also writes out whole.
# Some seeds produce a flood that reaches the producers in a less telling order;
# worth playing around with.
truth_seed = 20220906
truth_corr_length = 300.0
truth_sor = 0.2

# Localization radius, in metres, written on each observation for ert's DISTANCE
# strategy. 250 m is the best of 60-400 m on this case (ref the README's sweep).
localization_radius = 250.0
