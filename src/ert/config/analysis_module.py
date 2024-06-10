import logging
import math
from typing import Optional, Union

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from typing_extensions import Annotated, Literal

logger = logging.getLogger(__name__)


DEFAULT_IES_MAX_STEPLENGTH = 0.60
DEFAULT_IES_MIN_STEPLENGTH = 0.30
DEFAULT_IES_DEC_STEPLENGTH = 2.50
DEFAULT_ENKF_TRUNCATION = 0.98
DEFAULT_LOCALIZATION = False


class BaseSettings(BaseModel):
    enkf_truncation: Annotated[
        float,
        Field(gt=0.0, le=1.0, title="Singular value truncation"),
    ] = DEFAULT_ENKF_TRUNCATION

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def _lower(v: str) -> str:
    return v.lower()


InversionTypeES = Annotated[Literal["exact", "subspace"], BeforeValidator(_lower)]
es_description = """
    The type of inversion used in the algorithm. Every inversion method
    scales the variables. The options are:

    * `exact`:
        Computes an exact inversion which uses a Cholesky factorization in the
        case of symmetric, positive definite matrices.
    * `subspace`:
        This is an approximate solution. The approximation is that when
        U, w, V.T = svd(D_delta) then we assume that U @ U.T = I.
    """


InversionTypeIES = Annotated[
    Literal["direct", "subspace_exact", "subspace_projected"], BeforeValidator(_lower)
]
ies_description = """
    The type of inversion used in the algorithm. Every inversion method
    scales the variables. The options are:

    * `direct`:
        Solve directly, which involves inverting a matrix
        of shape (num_observations, num_observations).
    * `subspace_exact` :
        Use the Woodbury lemma to invert a matrix of
        size (ensemble_size, ensemble_size).
    * `subspace_projected` :
        Invert by projecting the covariance onto S.
    """


class ESSettings(BaseSettings):
    inversion: Annotated[
        InversionTypeES, Field(title="Inversion algorithm", description=es_description)
    ] = "exact"
    localization: Annotated[bool, Field(title="Adaptive localization")] = False
    localization_correlation_threshold: Annotated[
        Optional[float],
        Field(
            ge=0.0,
            le=1.0,
            title="Adaptive localization correlation threshold",
        ),
    ] = None

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


class IESSettings(BaseSettings):
    """A good start is max steplength of 0.6, min steplength of 0.3, and decline of 2.5",
    A steplength of 1.0 and one iteration results in ES update"""

    inversion: Annotated[
        InversionTypeIES,
        Field(title="Inversion algorithm", description=ies_description),
    ] = "subspace_exact"
    ies_max_steplength: Annotated[
        float,
        Field(ge=0.1, le=1.0, title="Gauss–Newton maximum steplength"),
    ] = DEFAULT_IES_MAX_STEPLENGTH
    ies_min_steplength: Annotated[
        float,
        Field(ge=0.1, le=1.0, title="Gauss–Newton minimum steplength"),
    ] = DEFAULT_IES_MIN_STEPLENGTH
    ies_dec_steplength: Annotated[
        float,
        Field(ge=1.1, le=10.0, title="Gauss–Newton steplength decline"),
    ] = DEFAULT_IES_DEC_STEPLENGTH


AnalysisModule = Union[ESSettings, IESSettings]
