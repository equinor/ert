import numpy as np
from pydantic import BaseModel, Field, field_validator


class InputConstraintConfig(BaseModel, extra="forbid"):
    weights: dict[str, float] = Field(
        examples=[
            {
                "weights": [
                    {"point_3D.x.0": 0},
                    {"point_3D.y.1": 0},
                    {"point_3D.z.2": 1},
                ],
                "upper_bound": 0.2,
            }
        ],
        description="""If we are trying to constrain only one control (i.e the z
    control) value, with an upper bound of 0.2, only control values (x, y, z) that
    satisfy the following equation will be allowed: `x-0 * 0 + y-1 * 0 + z-2 * 1  > 0.2`
""",
    )
    target: float | None = Field(
        default=None,
        description="""Only control values that satisfy the following
        equation will be allowed: sum of (<control> * weight) = target
        """,
    )
    lower_bound: float = Field(
        default=-np.inf,
        description="""Only control values that satisfy the following
        equation will be allowed: sum of (<control> * weight) >= lower_bound`
""",
    )
    upper_bound: float = Field(
        default=np.inf,
        description="""Only control values that satisfy the following
         equation will be allowed: sum of (<control> * weight) <= upper_bound`
        """,
    )

    @field_validator("weights")
    @classmethod
    def validate_weights_not_empty(cls, weights: dict[str, float]) -> dict[str, float]:
        if weights is None or weights == {}:
            raise ValueError("Input weight data required for input constraints")
        return weights
