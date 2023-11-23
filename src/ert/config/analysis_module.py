import logging
import math
from typing import TYPE_CHECKING, Optional, Type, TypedDict, Union

from pydantic import BaseModel, Extra, Field
from typing_extensions import Annotated

logger = logging.getLogger(__name__)


if TYPE_CHECKING:

    class VariableInfo(TypedDict):
        type: Union[Type[float], Type[int]]
        min: float
        value: Union[float, int]
        max: float
        step: float
        labelname: str


DEFAULT_IES_MAX_STEPLENGTH = 0.60
DEFAULT_IES_MIN_STEPLENGTH = 0.30
DEFAULT_IES_DEC_STEPLENGTH = 2.50
DEFAULT_ENKF_TRUNCATION = 0.98
DEFAULT_IES_INVERSION = 0
DEFAULT_LOCALIZATION = False


class BaseSettings(BaseModel):
    ies_inversion: Annotated[
        int, Field(strict=True, ge=0, le=3, title="Inversion algorithm")
    ] = DEFAULT_IES_INVERSION
    enkf_truncation: Annotated[
        float,
        Field(strict=True, gt=0.0, le=1.0, title="Singular value truncation"),
    ] = DEFAULT_ENKF_TRUNCATION

    class Config:
        extra = Extra.forbid
        validate_assignment = True


class ESSettings(BaseSettings):
    localization: Annotated[bool, Field(title="Adaptive localization")] = False
    localization_correlation_threshold: Annotated[
        Optional[float],
        Field(
            strict=True,
            ge=0.0,
            le=1.0,
            title="Adaptive localization correlation threshold",
        ),
    ] = None

    def correlation_threshold(self, ensemble_size: int) -> float:
        """Decides whether or not to use user-defined or default threshold.

        Default threshold taken from luo2022,
        Continuous Hyper-parameter OPtimization (CHOP) in an ensemble Kalman filter
        Section 2.3 - Localization in the CHOP problem
        """
        if self.localization_correlation_threshold is None:
            return 3 / math.sqrt(ensemble_size)
        else:
            return self.localization_correlation_threshold


class IESSettings(BaseSettings):
    """A good start is max steplength of 0.6, min steplength of 0.3, and decline of 2.5",
    A steplength of 1.0 and one iteration results in ES update"""

    ies_max_steplength: Annotated[
        float,
        Field(strict=True, ge=0.1, le=1.0, title="Gauss–Newton maximum steplength"),
    ] = DEFAULT_IES_MAX_STEPLENGTH
    ies_min_steplength: Annotated[
        float,
        Field(strict=True, ge=0.1, le=1.0, title="Gauss–Newton minimum steplength"),
    ] = DEFAULT_IES_MIN_STEPLENGTH
    ies_dec_steplength: Annotated[
        float,
        Field(strict=True, ge=1.1, le=10.0, title="Gauss–Newton steplength decline"),
    ] = DEFAULT_IES_DEC_STEPLENGTH


AnalysisModule = Union[ESSettings, IESSettings]
