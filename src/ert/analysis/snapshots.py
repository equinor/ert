from enum import Enum, auto
from typing import List

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
