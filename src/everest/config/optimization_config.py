import logging
from typing import Any, Self

from pydantic import BaseModel, Field, model_validator

from everest.config.cvar_config import CVaRConfig
from everest.optimizer.utils import get_ropt_plugin_manager
from everest.strings import EVEREST


class OptimizationConfig(BaseModel, extra="forbid"):
    algorithm: str = Field(
        default="optpp_q_newton",
        description="""Algorithm used by Everest.  Defaults to
optpp_q_newton, a quasi-Newton algorithm in Dakota's OPT PP library.
""",
    )
    convergence_tolerance: float | None = Field(
        default=None,
        description="""Defines the threshold value on relative change
in the objective function that indicates convergence.

The convergence_tolerance specification provides a real value for controlling
the termination of iteration.  In most cases, it is a relative convergence tolerance
for the objective function; i.e., if the change in the objective function between
successive iterations divided by the previous objective function is less than
the amount specified by convergence_tolerance, then this convergence criterion is
satisfied on the current iteration.

Since no progress may be made on one iteration followed by significant progress
on a subsequent iteration, some libraries require that the convergence tolerance
be satisfied on two or more consecutive iterations prior to termination of
iteration.

(From the Dakota Manual.)""",
    )
    backend: str | None = Field(
        default=None,
        description="""(deprecated) The optimization backend used.".

Currently, backends are included to use Dakota or SciPy ("dakota" and "scipy").
The Dakota backend is the default, and can be assumed to be installed. The SciPy
backend is optional, and will only be available if SciPy is installed on the
system.""",
    )
    backend_options: dict[str, Any] | None = Field(
        default=None,
        description="""Dict of optional parameters for the optimizer backend.
This dict of values is passed unchanged to the selected algorithm in the backend.

Note that the default Dakota backend ignores this option, because it requires a
list of strings rather than a dictionary. For setting Dakota backend options, see
the 'option' keyword.""",
    )
    constraint_tolerance: float | None = Field(
        default=None,
        description="""Determines the maximum allowable value of
infeasibility that any constraint in an optimization problem may possess and
still be considered to be satisfied.

It is specified as a positive real value.  If a constraint function is greater
than this value then it is considered to be violated by the optimization
algorithm.  This specification gives some control over how tightly the
constraints will be satisfied at convergence of the algorithm.  However, if the
value is set too small the algorithm may terminate with one or more constraints
being violated.

(From the Dakota Manual.)""",
    )
    cvar: CVaRConfig | None = Field(
        default=None,
        description="""Directs the optimizer to use CVaR estimation.

When this section is present Everest will use Conditional Value at Risk (CVaR)
to minimize risk. Effectively this means that at each iteration the objective
and constraint functions will be calculated as the mean over the sub-set of the
realizations that perform worst. The size of this set is specified as an
absolute number or as a percentile value. These options are selected by setting
either the **number_of_realizations** option, or the **percentile** option,
which are mutually exclusive.
""",
    )
    max_batch_num: int | None = Field(
        default=None,
        gt=0,
        description="""Limits the number of batches of simulations
during optimization, where 0 represents unlimited simulation batches.
When max_batch_num is specified and the current batch index is greater than
max_batch_num an exception is raised.""",
    )
    max_function_evaluations: int | None = Field(
        default=None,
        gt=0,
        description="""Limits the maximum number of function evaluations.

The max_function_evaluations controls the number of control update steps the optimizer
will allow before convergence is obtained.

See max_iterations for a description.
""",
    )
    max_iterations: int | None = Field(
        default=None,
        gt=0,
        description="""Limits the maximum number of iterations.

The difference between an iteration and a batch is that an iteration corresponds to
a complete accepted batch (i.e., a batch that provides an improvement in the
objective function while satisfying all constraints).""",
    )
    min_pert_success: int | None = Field(
        default=None,
        gt=0,
        description="""specifies the minimum number of successfully completed
evaluations of perturbed controls required to compute a gradient.  The optimization
process will stop if this minimum is not reached, and otherwise a gradient will be
computed based on the set of successful perturbation runs.  The minimum is checked for
each realization individually.

A special case is robust optimization with `perturbation_num: 1`.  In that case the
minimum applies to all realizations combined. In other words, a robust gradient may then
still be computed based on a subset of the realizations.

The user-provided value is reset to perturbation_num if it is larger than this number
and a message is produced.  In the special case of robust optimization case with
`perturbation_num: 1` the maximum allowed value is the number of realizations specified
by realizations instead.""",
    )
    min_realizations_success: int | None = Field(
        default=None,
        ge=0,
        description="""Minimum number of realizations

The minimum number of realizations that should be available for the computation
of either expected function values (both objective function and constraint
functions) or of the expected gradient.  Note that this keyword does not apply
to gradient computation in the robust case with 1 perturbation in which the
expected gradient is computed directly.

The optimization process will stop if this minimum is not reached, and otherwise
the expected objective function value (and expected gradient/constraint function
values) will be computed based on the set of successful contributions.  In other
words, a robust objective function, a robust gradient and robust constraint
functions may then still be computed based on a subset of the realizations.

The user-provided value is reset to the number of realizations specified by
realizations if it is larger than this number and a message is produced.

Note that it is possible to set the minimum number of successful realizations equal
to zero. Some optimization algorithms are able to handle this and will proceed even
if all realizations failed. Most algorithms are not capable of this and will adjust
the value to be equal to one.
""",
    )
    options: list[str] | dict[str, Any] | None = Field(
        default=None,
        description="""specifies non-validated, optional
passthrough parameters for the optimizer

| Examples used are
| - max_repetitions = 300
| - retry_if_fail
| - classical_search 1""",
    )
    perturbation_num: int | None = Field(
        default=None,
        gt=0,
        description="""The number of perturbed control vectors per realization.

The number of simulation runs used for estimating the gradient is equal to the
the product of perturbation_num and model.realizations.""",
    )
    speculative: bool = Field(
        default=False,
        description="""specifies whether to enable speculative computation.

The speculative specification enables speculative computation of gradient and/or
Hessian information, where applicable, for parallel optimization studies. By
speculating that the derivative information at the current point will be used
later, the complete data set (all available gradient/Hessian information) can be
computed on every function evaluation. While some of these computations will be
wasted, the positive effects are a consistent parallel load balance and usually
shorter wall clock time. The speculative specification is applicable only when
parallelism in the gradient calculations can be exploited by Dakota (it will be
ignored for vendor numerical gradients).  (From the Dakota Manual.)""",
    )
    parallel: bool = Field(
        default=True,
        description="""whether to allow parallel function evaluation.

By default Everest will evaluate a single function and gradient evaluation at
a time. In case of gradient-free optimizer this can be highly inefficient,
since these tend to need many independent function evaluations at each
iteration. By setting parallel to True, multiple functions may be evaluated in
parallel, if supported by the optimization algorithm.

The default is to use parallel evaluation if supported.
""",
    )

    _optimization_plugin: str

    @model_validator(mode="after")
    def validate_backend_and_algorithm(self) -> Self:
        if self.backend is not None:
            message = (
                "optimization.backend is deprecated. "
                "The correct backend will be inferred by the algorithm. "
                "If several backends have an algorithm named A, you need to pick "
                "a specific backend B by putting B/A in optimization.algorithm."
            )
            print(message)
            logging.getLogger(EVEREST).warning(message)

        if self.backend_options is not None:
            message = (
                "optimization.backend_options is deprecated. "
                "Please use optimization.options instead, "
                "it will accept both objects and lists of strings."
            )
            print(message)
            logging.getLogger(EVEREST).warning(message)

        # Update the default for backends that are not dakota:
        if (
            self.backend not in {None, "dakota"}
            and "algorithm" not in self.model_fields_set
        ):
            self.algorithm = "default"

        algorithm = (  # Do not modify self.algorithm yet, still needed.
            self.algorithm
            if self.backend is None
            else f"{self.backend}/{self.algorithm}"
        )

        plugin_manager = get_ropt_plugin_manager()
        plugin_name = plugin_manager.get_plugin_name("optimizer", algorithm)
        if plugin_name is None:
            raise ValueError(f"Optimizer algorithm '{algorithm}' not found")
        self._optimization_plugin_name = plugin_name

        plugin_manager.get_plugin("optimizer", algorithm).validate_options(
            self.algorithm, self.options or self.backend_options
        )

        self.backend = None
        self.algorithm = algorithm

        return self

    @property
    def optimization_plugin_name(self) -> str:
        return self._optimization_plugin_name
