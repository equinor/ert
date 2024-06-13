from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class RestartConfig(BaseModel):  # type: ignore
    max_restarts: Optional[int] = Field(
        default=None,
        gt=0,
        description="""The maximum number of restarts.

Sets the maximum number of times that the optimization process will be
restarted.

The default is equal to a single restart.
""",
    )

    restart_from: Literal["initial", "last", "optimal", "last_optimal"] = Field(
        description="""Restart from the initial, optimal or the last controls.

When restarting, the initial values for the new run are set according to this field:
- initial: Use the initial controls from the configuration
- last: Use the last controls used by the previous run
- optimal: Use the controls from the optimal solution found so far
- last_optimal: Use the controls from the optimal solution found in previous run

When restarting from optimal values, the best result obtained so far (either
overall, or in the last restart run) is used, which is defined as the result
with the maximal weighted total objective value. If the `constraint_tolerance`
option is set in the `optimization` section, this tolerance will be used to
exclude results that violate a constraint.

""",
    )

    model_config = ConfigDict(
        extra="forbid",
    )
