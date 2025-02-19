from pydantic import BaseModel, Field, field_validator


class InputConstraintConfig(BaseModel, extra="forbid"):
    weights: dict[str, float] = Field(
        description="""**Example**
If we are trying to constrain only one control (i.e the z control) value:
| input_constraints:
| - weights:
|   point_3D.x.0: 0
|   point_3D.y.1: 0
|   point_3D.z.2: 1
| upper_bound: 0.2

Only control values (x, y, z) that satisfy the following equation will be allowed:
`x-0 * 0 + y-1 * 0 + z-2 * 1  > 0.2`
""",
    )
    target: float | None = Field(
        default=None,
        description="""**Example**
| input_constraints:
| - weights:
|   point_3D.x.0: 1
|   point_3D.y.1: 2
|   point_3D.z.2: 3
| target: 4

Only control values (x, y, z) that satisfy the following equation will be allowed:
`x-0 * 1 + y-1 * 2 + z-2 * 3  = 4`
""",
    )
    lower_bound: float | None = Field(
        default=None,
        description="""**Example**
| input_constraints:
| - weights:
|   point_3D.x.0: 1
|   point_3D.y.1: 2
|   point_3D.z.2: 3
| lower_bound: 4

Only control values (x, y, z) that satisfy the following
equation will be allowed:
`x-0 * 1 + y-1 * 2 + z-2 * 3  >= 4`
""",
    )
    upper_bound: float | None = Field(
        default=None,
        description="""**Example**
| input_constraints:
| - weights:
|   point_3D.x.0: 1
|   point_3D.y.1: 2
|   point_3D.z.2: 3
| upper_bound: 4

Only control values (x, y, z) that satisfy the following equation will be allowed:
`x-0 * 1 + y-1 * 2 + z-2 * 3  <= 4`
""",
    )

    @field_validator("weights")
    @classmethod
    # pylint: disable=E0213
    def validate_weights_not_empty(cls, weights: dict[str, float]) -> dict[str, float]:
        if weights is None or weights == {}:
            raise ValueError("Input weight data required for input constraints")
        return weights
