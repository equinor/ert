from __future__ import annotations

import logging
import math
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)


DEFAULT_ENKF_TRUNCATION = 0.98
DEFAULT_LOCALIZATION = False


def _upper(v: str) -> str:
    return v.upper()


InversionTypeES = Annotated[Literal["EXACT", "SUBSPACE"], BeforeValidator(_upper)]
es_description = """
    The type of inversion used in the algorithm. Every inversion method
    scales the variables. The options are:

    * EXACT:
        Computes an exact inversion which uses a Cholesky factorization in the
        case of symmetric, positive definite matrices.
    * SUBSPACE:
        This is an approximate solution. The approximation is that when
        U, w, V.T = svd(D_delta) then we assume that U @ U.T = I.
    """

loc_description = """
    The default adaptive localization correlation threshold
    is computed as 3/sqrt(ensemble_size), where ensemble_size
    is the number of active realizations in the ensemble.

    You can override this value by setting a custom threshold here or in the config.
    """


class ESSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    enkf_truncation: Annotated[
        float,
        Field(gt=0.0, le=1.0, title="Singular value truncation"),
    ] = DEFAULT_ENKF_TRUNCATION
    inversion: Annotated[
        InversionTypeES, Field(title="Inversion algorithm", description=es_description)
    ] = "EXACT"
    localization: Annotated[
        bool, Field(title="Enable adaptive localization", description=loc_description)
    ] = False
    custom_localization_threshold: Annotated[
        bool, Field(title="Custom localization correlation threshold")
    ] = False
    localization_correlation_threshold: Annotated[
        float | None,
        Field(
            ge=0.0,
            le=1.0,
            title="Custom adaptive localization correlation threshold",
        ),
    ] = None

    @model_validator(mode="after")
    def _validate_custom_localization_threshold(self) -> ESSettings:
        if (
            self.localization_correlation_threshold is not None
            and self.custom_localization_threshold is False
        ):
            self.custom_localization_threshold = True
        return self

    def correlation_threshold(self, ensemble_size: int) -> float:
        """Decides whether to use user-defined or default threshold.

        Default threshold taken from luo2022,
        Continuous Hyper-parameter Optimization (CHOP) in an ensemble Kalman filter
        Section 2.3 - Localization in the CHOP problem
        """
        if self.localization_correlation_threshold is None:
            return 3 / math.sqrt(ensemble_size)
        else:
            return self.localization_correlation_threshold


AnalysisModule = ESSettings
