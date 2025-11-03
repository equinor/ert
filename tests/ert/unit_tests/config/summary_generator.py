import hypothesis.strategies as st
import numpy as np
from hypothesis.extra.numpy import from_dtype
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

positive_floats = from_dtype(
    np.dtype(np.float32),
    min_value=np.float32(0.1),
    max_value=np.float32(1e19),
    allow_nan=False,
    allow_infinity=False,
)


@st.composite
def unsmrys(
    draw,
    num_params,
    report_steps,
    mini_steps,
    days,
):
    """
    Simplified strategy for unsmrys that will generate a summary file
    with one ministep per report step.
    """
    n = num_params
    rs = sorted(draw(report_steps))
    ms = sorted(draw(mini_steps), reverse=True)
    ds = sorted(draw(days), reverse=True)
    steps = []
    for r in rs:
        minis = [
            SummaryMiniStep(
                ms.pop(),
                [ds.pop(), *draw(st.lists(positive_floats, min_size=n, max_size=n))],
            )
        ]
        steps.append(SummaryStep(r, minis))
    return Unsmry(steps)


def simple_unsmry():
    return Unsmry(
        steps=[
            SummaryStep(
                seqnum=0,
                ministeps=[
                    SummaryMiniStep(mini_step=0, params=[0.0, 5.629901e16]),
                ],
            )
        ]
    )


def simple_smspec():
    return Smspec(
        nx=2,
        ny=2,
        nz=2,
        restarted_from_step=0,
        num_keywords=2,
        restart="        ",
        keywords=["TIME    ", "FOPR"],
        well_names=[":+:+:+:+", "        "],
        region_numbers=[-32676, 0],
        units=["HOURS   ", "SM3"],
        start_date=Date(day=1, month=1, year=2014, hour=0, minutes=0, micro_seconds=0),
        intehead=SmspecIntehead(
            unit=UnitSystem.METRIC,
            simulator=Simulator.ECLIPSE_100,
        ),
    )
