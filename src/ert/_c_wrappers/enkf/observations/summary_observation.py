from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SummaryObservation:
    summary_key: str
    observation_key: str
    value: float
    std: float
    std_scaling: float = 1.0
