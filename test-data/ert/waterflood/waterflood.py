#!/usr/bin/env python3
"""Two-phase (water/oil) reservoir simulation to use as a forward model.

Runs the waterflood of `definition.py` with `minires`, a small two-point
flux approximation simulator, and writes the well responses as summary files.

At the first iteration the permeability field is sampled from its prior and
written to `permx.bgrdecl` (FORWARD_INIT), afterwards it is read back from the
file that ert has written from the updated ensemble.
"""

import datetime
import json
import sys
from pathlib import Path

import numpy as np
import resfo
from definition import (
    bhp_prod,
    ct,
    dt,
    dz,
    injectors,
    lx,
    nsteps,
    nx,
    ny,
    perm_mean,
    perm_std,
    poro,
    pressure_init,
    producers,
    rate_inj,
    start_date,
    summary_keys,
    swc,
    visc_oil,
    visc_water,
    well_radius,
)
from minires import Grid2D, ResSim
from minires.geostat import gaussian_fields
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

# Darcy's law needs no conversion factor in a coherent unit system; in the metric
# one (m, day, bar, mD, cP) it needs this constant, which is ECLIPSE's CDARCY.
CDARCY = 0.008527

# Units of the summary vectors. The model has no PVT, so reservoir and standard
# volumes are the same thing here; and its rates are areal (per unit thickness)
# until `simulate` multiplies them by the thickness of the reservoir.
SUMMARY_UNITS = {"WWCT": "        ", "WOPR": "SM3/DAY ", "WBHP": "BARSA   "}


def sample_prior_permeability(corr_length: float, rng: np.random.Generator):
    """Draw a log-normal permeability field with a Gaussian variogram."""
    mesh = Grid2D(Lx=lx, Ly=lx, Nx=nx, Ny=ny).mesh
    field = gaussian_fields(mesh, 1, corr_length, rng)[0]
    return np.exp(np.log(perm_mean) + perm_std * field).reshape(nx, ny)


def make_model(permx, sor: float) -> ResSim:
    """The reservoir model of `definition.py`, for a given permeability and `sor`."""
    wells = {
        well.name: {
            "xy": [well.east, well.north],
            "rate": rate_inj / len(injectors) / dz,
            "rw": well_radius,
        }
        for well in injectors
    }
    for well in producers:
        wells[well.name] = {
            "xy": [well.east, well.north],
            "bhp": bhp_prod,
            "rw": well_radius,
        }

    return ResSim(
        Lx=lx,
        Ly=lx,
        Nx=nx,
        Ny=ny,
        cdarcy=CDARCY,
        K=permx,
        por=np.full((nx, ny), poro),
        ct=ct,
        # Quadratic Corey curves; the residual oil saturation is uncertain
        fluid={
            "vw": visc_water,
            "vo": visc_oil,
            "swc": swc,
            "sor": sor,
            "nw": 2,
            "no": 2,
        },
        wells=wells,
    )


def simulate(model: ResSim) -> dict[str, list[float]]:
    """Run the flood, and report the wells as summary vectors.

    The model is areal, so its rates are per unit thickness (ref `ResSim.cdarcy`)
    and are multiplied back up by the thickness of the reservoir here.
    """
    saturation, _ = model.sim(
        dt,
        nsteps,
        S0=np.full(model.Nxy, swc),
        P0=np.full(model.Nxy, pressure_init),
        pbar=False,
    )

    cells = [model.xy2ind(well.east, well.north) for well in producers]
    # The saturation at the end of each report step, hence `[1:]`
    water_cut = np.array(
        [model.fluid.fractional_flow(s)[cells] for s in saturation[1:]]
    )
    # Produced, hence positive; the injectors are the first rows of the well arrays
    rates = -model.wells.actual_rates[len(injectors) :] * dz
    oil_rate = rates * (1 - water_cut.T)

    return dict(
        zip(
            summary_keys,
            [
                *water_cut.T.tolist(),
                *oil_rate.tolist(),
                *model.wells.actual_bhp[: len(injectors)].tolist(),
            ],
            strict=True,
        )
    )


def write_summary(vectors: dict[str, list[float]], eclbase: str) -> None:
    """Write the summary vectors as an ECLIPSE-style .SMSPEC/.UNSMRY pair."""
    keys = list(vectors)
    keywords = [key.split(":")[0] for key in keys]

    smspec = Smspec(
        nx=nx,
        ny=ny,
        nz=1,
        restarted_from_step=0,
        num_keywords=1 + len(keys),
        restart="        ",
        keywords=["TIME    ", *keywords],
        well_names=[":+:+:+:+", *[key.split(":")[1] for key in keys]],
        region_numbers=[-32676, *([0] * len(keys))],
        units=["DAYS    ", *[SUMMARY_UNITS[kw] for kw in keywords]],
        start_date=Date.from_datetime(datetime.datetime.fromisoformat(start_date)),
        intehead=SmspecIntehead(
            unit=UnitSystem.METRIC, simulator=Simulator.ECLIPSE_100
        ),
    )
    unsmry = Unsmry(
        steps=[
            SummaryStep(
                seqnum=step,
                ministeps=[
                    SummaryMiniStep(
                        mini_step=0,
                        params=[dt * (step + 1), *[vectors[key][step] for key in keys]],
                    )
                ],
            )
            for step in range(nsteps)
        ]
    )
    smspec.to_file(f"{eclbase}.SMSPEC")
    unsmry.to_file(f"{eclbase}.UNSMRY")


def load_parameters(filename: str) -> dict:
    with Path(filename).open(encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    iens = int(sys.argv[1])
    iteration = int(sys.argv[2])
    eclbase = sys.argv[3]
    rng = np.random.default_rng(iens)

    parameters = load_parameters("parameters.json")
    sor = float(parameters["SOR"]["value"])

    if iteration == 0:
        permx = sample_prior_permeability(
            corr_length=float(parameters["CORR_LENGTH"]["value"]), rng=rng
        )
        resfo.write(
            "permx.bgrdecl",
            [("PERMX   ", permx.flatten(order="F").astype(np.float32))],
        )
    else:
        permx = resfo.read("permx.bgrdecl")[0][1].reshape(nx, ny, order="F")

    # The update is made in log-space (INIT_TRANSFORM:LN, OUTPUT_TRANSFORM:EXP),
    # so the permeability is positive; clip all the same, as it must not be zero.
    permx = np.asarray(permx, dtype=float).clip(min=1e-3)

    write_summary(simulate(make_model(permx, sor)), eclbase)
