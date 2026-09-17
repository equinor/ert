from typing import Literal

from pydantic import BaseModel, ConfigDict


class PriorBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PriorConst(PriorBaseModel):
    """Constant parameter prior"""

    function: Literal["const"] = "const"
    value: float


class PriorTrig(PriorBaseModel):
    """Triangular distribution parameter prior"""

    function: Literal["trig"] = "trig"
    min: float
    max: float
    mode: float


class PriorNormal(PriorBaseModel):
    """Normal distribution parameter prior"""

    function: Literal["normal"] = "normal"
    mean: float
    std: float


class PriorLogNormal(PriorBaseModel):
    """Log-normal distribution parameter prior"""

    function: Literal["lognormal"] = "lognormal"
    mean: float
    std: float


class PriorErtTruncNormal(PriorBaseModel):
    """
    ERT Truncated normal distribution parameter prior

    ERT differs from the usual distribution by that it simply clamps on `min`
    and `max`, which gives a bias towards the extremes.

    """

    function: Literal["ert_truncnormal"] = "ert_truncnormal"
    mean: float
    std: float
    min: float
    max: float


class PriorStdNormal(PriorBaseModel):
    """
    Standard normal distribution parameter prior

    Normal distribution with mean of 0 and standard deviation of 1
    """

    function: Literal["stdnormal"] = "stdnormal"


class PriorUniform(PriorBaseModel):
    """Uniform distribution parameter prior"""

    function: Literal["uniform"] = "uniform"
    min: float
    max: float


class PriorErtDUniform(PriorBaseModel):
    """
    ERT Discrete uniform distribution parameter prior

    This discrete uniform distribution differs from the standard by using the
    `bins` parameter. Normally, `a`, and `b` are integers, and the sample space
    are the integers between. ERT allows `a` and `b` to be arbitrary floats,
    where the sample space is binned.

    """

    function: Literal["ert_duniform"] = "ert_duniform"
    bins: int
    min: float
    max: float


class PriorLogUniform(PriorBaseModel):
    """Logarithmic uniform distribution parameter prior"""

    function: Literal["loguniform"] = "loguniform"
    min: float
    max: float


class PriorErtErf(PriorBaseModel):
    """ERT Error function distribution parameter prior"""

    function: Literal["ert_erf"] = "ert_erf"
    min: float
    max: float
    skewness: float
    width: float


class PriorErtDErf(PriorBaseModel):
    """ERT Discrete error function distribution parameter prior"""

    function: Literal["ert_derf"] = "ert_derf"
    bins: int
    min: float
    max: float
    skewness: float
    width: float


Prior = (
    PriorConst
    | PriorTrig
    | PriorNormal
    | PriorLogNormal
    | PriorErtTruncNormal
    | PriorStdNormal
    | PriorUniform
    | PriorErtDUniform
    | PriorLogUniform
    | PriorErtErf
    | PriorErtDErf
)
