from enum import Enum, auto
from typing import Any, Dict, List

import numpy as np
from pydantic import BaseModel


class ObservationStatus(Enum):
    ACTIVE = auto()
    MISSING_RESPONSE = auto()
    OUTLIER = auto()
    STD_CUTOFF = auto()


class ObservationAndResponseSnapshot(BaseModel):
    obs_name: str
    obs_val: float
    obs_std: float
    obs_scaling: float
    response_mean: float
    response_std: float
    response_mean_mask: bool
    response_std_mask: bool
    index: str

    @property
    def status(self) -> ObservationStatus:
        if np.isnan(self.response_mean):
            return ObservationStatus.MISSING_RESPONSE
        elif not self.response_std_mask:
            return ObservationStatus.STD_CUTOFF
        elif not self.response_mean_mask:
            return ObservationStatus.OUTLIER
        return ObservationStatus.ACTIVE

    def get_status(self) -> str:
        if self.status == ObservationStatus.MISSING_RESPONSE:
            return "Deactivated, missing response(es)"
        if self.status == ObservationStatus.STD_CUTOFF:
            return f"Deactivated, ensemble std ({self.response_std:.3f}) > STD_CUTOFF"
        if self.status == ObservationStatus.OUTLIER:
            return "Deactivated, outlier"
        return "Active"


class SmootherSnapshot(BaseModel):
    source_ensemble_name: str
    target_ensemble_name: str
    alpha: float
    std_cutoff: float
    global_scaling: float
    update_step_snapshots: List[ObservationAndResponseSnapshot]

    @property
    def header(self) -> List[str]:
        return [
            "Observation name",
            "Index",
            "Obs value",
            "Obs std",
            "Obs scaling",
            "Scaled std",
            "Response mean",
            "Response std",
            "Status",
        ]

    @property
    def csv(self) -> List[List[Any]]:
        data = []
        for step in self.update_step_snapshots:
            data.append(
                [
                    step.obs_name,
                    step.index,
                    step.obs_val,
                    step.obs_std,
                    step.obs_scaling,
                    step.obs_scaling * step.obs_std,
                    step.response_mean,
                    step.response_std,
                    step.get_status(),
                ]
            )
        return data

    @property
    def extra(self) -> Dict[str, str]:
        return {
            "Parent ensemble": self.source_ensemble_name,
            "Target ensemble": self.target_ensemble_name,
            "Alpha": str(self.alpha),
            "Global scaling": str(self.global_scaling),
            "Standard cutoff": str(self.std_cutoff),
            "Active observations": str(
                sum(
                    val.status == ObservationStatus.ACTIVE
                    for val in self.update_step_snapshots
                )
            ),
            "Deactivated observations - missing respons(es)": str(
                sum(
                    val.status == ObservationStatus.MISSING_RESPONSE
                    for val in self.update_step_snapshots
                )
            ),
            "Deactivated observations - ensemble_std > STD_CUTOFF": str(
                sum(
                    val.status == ObservationStatus.STD_CUTOFF
                    for val in self.update_step_snapshots
                )
            ),
            "Deactivated observations - outliers": str(
                sum(
                    val.status == ObservationStatus.OUTLIER
                    for val in self.update_step_snapshots
                )
            ),
        }
