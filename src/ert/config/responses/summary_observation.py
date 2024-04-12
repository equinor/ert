from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SummaryObservation:
    summary_key: str
    observation_key: str
    value: float
    std: float
    std_scaling: float = 1.0

    def __post_init__(self) -> None:
        if self.std <= 0:
            raise ValueError("Observation uncertainty must be strictly > 0")
