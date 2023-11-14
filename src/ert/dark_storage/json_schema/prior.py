import sys
from typing import Union

from pydantic import BaseModel

if sys.version_info < (3, 8):
    from typing_extensions import Literal
else:
    from typing import Literal


class PriorConst(BaseModel):
    """
    Constant parameter prior
    """

    function: Literal["const"] = "const"
    value: float


class PriorTrig(BaseModel):
    """
    Triangular distribution parameter prior
    """

    function: Literal["trig"] = "trig"
    min: float
    max: float
    mode: float


class PriorNormal(BaseModel):
    """
    Normal distribution parameter prior
    """

    function: Literal["normal"] = "normal"
    mean: float
    std: float


class PriorLogNormal(BaseModel):
    """
    Log-normal distribution parameter prior
    """

    function: Literal["lognormal"] = "lognormal"
    mean: float
    std: float


class PriorErtTruncNormal(BaseModel):
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


class PriorStdNormal(BaseModel):
    """
    Standard normal distribution parameter prior

    Normal distribution with mean of 0 and standard deviation of 1
    """

    function: Literal["stdnormal"] = "stdnormal"


class PriorUniform(BaseModel):
    """
    Uniform distribution parameter prior
    """

    function: Literal["uniform"] = "uniform"
    min: float
    max: float


class PriorErtDUniform(BaseModel):
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


class PriorLogUniform(BaseModel):
    """
    Logarithmic uniform distribution parameter prior
    """

    function: Literal["loguniform"] = "loguniform"
    min: float
    max: float


class PriorErtErf(BaseModel):
    """
    ERT Error function distribution parameter prior
    """

    function: Literal["ert_erf"] = "ert_erf"
    min: float
    max: float
    skewness: float
    width: float


class PriorErtDErf(BaseModel):
    """
    ERT Discrete error function distribution parameter prior
    """

    function: Literal["ert_derf"] = "ert_derf"
    bins: int
    min: float
    max: float
    skewness: float
    width: float


Prior = Union[
    PriorConst,
    PriorTrig,
    PriorNormal,
    PriorLogNormal,
    PriorErtTruncNormal,
    PriorStdNormal,
    PriorUniform,
    PriorErtDUniform,
    PriorLogUniform,
    PriorErtErf,
    PriorErtDErf,
]
