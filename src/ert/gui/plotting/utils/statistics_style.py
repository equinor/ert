from __future__ import annotations

from typing import NamedTuple


class StatisticStyle(NamedTuple):
    label: str
    line_style: str
    is_band: bool


STATISTICS = {
    "mean": StatisticStyle("Mean", "-", is_band=False),
    "p50": StatisticStyle("P50", "--", is_band=False),
    "std": StatisticStyle("Std dev", ":", is_band=True),
    "min-max": StatisticStyle("Min/Max", ":", is_band=True),
    "p10-p90": StatisticStyle("P10-P90", "--", is_band=True),
    "p33-p67": StatisticStyle("P33-P67", "-.", is_band=True),
}

BAND_AREA_STYLE = "#"

DEFAULT_ENABLED_STATISTICS = {"mean", "p10-p90"}
