from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import polars as pl


class ObservationStatus(StrEnum):
    ACTIVE = "Active"
    MISSING_RESPONSE = "nan"
    OUTLIER = "Deactivated, outlier"
    STD_CUTOFF = "collapsed"


@dataclass
class SmootherSnapshot:
    source_ensemble_name: str
    target_ensemble_name: str
    alpha: float
    std_cutoff: float
    global_scaling: float
    observations_and_responses: pl.DataFrame | None = None

    @property
    def header(self) -> list[str]:
        assert self.observations_and_responses is not None
        return self.observations_and_responses.columns

    @property
    def csv(self) -> list[tuple[Any]]:
        assert self.observations_and_responses is not None
        return self.observations_and_responses.rows()

    @property
    def extra(self) -> dict[str, str]:
        assert self.observations_and_responses is not None
        return {
            "Parent ensemble": self.source_ensemble_name,
            "Target ensemble": self.target_ensemble_name,
            "Alpha": str(self.alpha),
            "Global scaling": str(self.global_scaling),
            "Standard cutoff": str(self.std_cutoff),
            "Active observations": str(
                self.observations_and_responses.filter(
                    pl.col("status") == ObservationStatus.ACTIVE
                ).height
            ),
            "Deactivated observations - missing respons(es)": str(
                self.observations_and_responses.filter(
                    pl.col("status") == ObservationStatus.MISSING_RESPONSE
                ).height
            ),
            "Deactivated observations - ensemble_std > STD_CUTOFF": str(
                self.observations_and_responses.filter(
                    pl.col("status") == ObservationStatus.STD_CUTOFF
                ).height
            ),
            "Deactivated observations - outliers": str(
                self.observations_and_responses.filter(
                    pl.col("status") == ObservationStatus.OUTLIER
                ).height
            ),
        }
