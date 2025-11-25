import importlib
import logging
from textwrap import dedent
from typing import Any, Self

from pydantic import BaseModel, Field, model_validator
from ropt.workflow import find_optimizer_plugin, validate_optimizer_options

from everest.config.cvar_config import CVaRConfig
from everest.strings import EVEREST


class OptimizationConfig(BaseModel, extra="forbid"):
    algorithm: str = Field(
        default="optpp_q_newton",
        description=dedent(
            """
            The optimization algorithm used by Everest.  Defaults to
            `optpp_q_newton`, a quasi-Newton algorithm in Dakota's OPT++
            library.
            """
        ),
    )
    convergence_tolerance: float | None = Field(
        default=None,
        description=dedent(
            """
            Defines the threshold value to test for convergence.

            In most cases, this is used to set a relative convergence tolerance
            for the objective function: if the change in the objective function
            between one or more successive iterations divided by the previous
            objective function is less than `convergence_tolerance`, the
            optimization is terminated.

            Note: this option is passed to the optimization backend unchanged,
            and therefore its actual interpretation depends on the algorithm
            used.
            """
        ),
    )
    backend: str | None = Field(
        default=None,
        description=dedent(
            """
            [Deprecated]

            The correct backend will be inferred by the method. If several backends
            have a method named `A`, pick a specific backend `B` by putting `B/A` in
            the `method` field.
            """
        ),
    )
    backend_options: dict[str, Any] | None = Field(
        default=None,
        description=dedent(
            """
            [Deprecated]

            This field is deprecated. Please use `options` instead, it will
            accept both objects and lists of strings.
            """
        ),
    )
    options: list[str] | dict[str, Any] | None = Field(
        default=None,
        description=dedent(
            """
            Specify options that are passed unchanged to the optimization
            algorithm.

            This field  accepts dictionaries with key/values pair or lists of
            strings. The type required depends on the optimization backend that
            is used. The Dakota backend requires a list of a strings that are
            added to the Dakota configuration. The SciPy backend requires a
            dictionary with values, which will be passed as keyword arguments to
            the optimizer.

            Consult the documentation of the optimization backend for valid
            options.
            """
        ),
    )
    constraint_tolerance: float | None = Field(
        default=None,
        description=dedent(
            """
            Output constraint tolerance for Dakota.

            Determines the maximum allowable value of infeasibility that any
            constraint in an optimization problem may possess and still be
            considered to be satisfied.

            It is specified as a positive real value.  If a constraint function
            is greater than this value then it is considered to be violated by
            the optimization algorithm.  This specification gives some control
            over how tightly the constraints will be satisfied at convergence of
            the algorithm.  However, if the value is set too small the algorithm
            may terminate with one or more constraints being violated.

            This option is only used by the Dakota backend.
            """
        ),
    )
    cvar: CVaRConfig | None = Field(
        default=None,
        description=dedent(
            """
            Directs the optimizer to use CVaR estimation.

            When this section is present, Conditional Value at Risk (CVaR) will
            be used to minimize risk. Effectively this means that at each
            iteration the objective and constraint functions will be calculated
            as the mean over the sub-set of the realizations that perform worst.
            The size of this set is specified as an absolute number or as a
            percentile value. These options are selected by setting either the
            `number_of_realizations` option, or the `percentile` option, which
            are mutually exclusive.
            """
        ),
    )
    max_batch_num: int | None = Field(
        default=None,
        gt=0,
        description=dedent(
            """
            Limit the number of batches of simulations.

            The optimization will be terminated if the given number of batches
            has been reached.
            """
        ),
    )
    max_function_evaluations: int | None = Field(
        default=None,
        gt=0,
        description=dedent(
            """
            Limits the maximum number of function evaluations.

            If the given number of function evaluations is reached, the
            optimization will be terminated. This differs from the
            `max_batch_num` and `max_iterations` options, because these may
            require multiple (or no) function evaluations.

            Note: Evaluations of perturbed values are not included in this
                  limit.
            """
        ),
    )
    max_iterations: int | None = Field(
        default=None,
        gt=0,
        description=dedent(
            """
            Limits the maximum number of iterations.

            This limits the number of iterations that the optimization algorithm
            will perform.

            This differs from `max_batch_num` and `max_functions` options, since
            an iteration may require multiple batches and/or function
            evaluations.

            This option is passed through unchanged, and its exact
            interpretation depends on the backend optimization algorithm used.
            """
        ),
    )
    min_pert_success: int | None = Field(
        default=None,
        gt=0,
        description=dedent(
            """
            The number of perturbations that must succeed.

            To calculate gradients, the optimizer requests a number of forward
            model evaluations for perturbed variables. There are two possible
            ways to calculate the gradient of the objective function:

            1. For each realization, `perturbation_num` objective functions are
               evaluated with perturbed controls, which will be used to
               calculate a gradient for that realization. The gradient
               calculation for a single realization is considered successful if
               at least `min_pert_success` forward model runs for that
               realization are successful. The overall gradient can be
               calculated from the set of realization gradients if a sufficient
               number succeeded (see the `min_realizations_success` option). If
               not, the optimization terminates with an error.
            2. If the `perturbation_num` field is equal to 1, the overall
               gradient is directly calculated from a single perturbation for
               each realization. In this case `min_pert_success` is internally
               set to 1, and the number of successful realizations is equal to
               the number of successful perturbed runs. The gradient calculation
               succeeds if this is at least equal to `min_realizations_success`.

            Note: The value of `min_pert_success` is internally adjusted to be
            capped by `perturbation_num`.
            """
        ),
    )
    min_realizations_success: int | None = Field(
        default=None,
        ge=0,
        description=dedent(
            """
            Minimum number of successful realizations.

            The overall objective functions and gradients are calculated from a
            set of forward model runs for a number of realizations of an
            ensemble. This is done by aggregating the functions and gradients,
            usually by averaging. If one or more of the forward model
            evaluations fail, this may lead to a failure of either the function
            or the gradient evaluation. However, the number of realizations that
            are successfully evaluated may still be sufficient to calculate
            robust aggregated functions and gradients.

            To calculate the aggregated objective function and gradient, the
            number of successful forward models that calculate the objective
            function for each realization must be at least equal to
            `min_realizations_success`. The number of realizations that are
            successful in terms of the gradient calculation is additionally
            determined by the `min_pert_success` field, see its documentation
            for more details.

            If these requirements are not met, the optimizer will stop and
            report an error.

            Note: The value of `min_realizations_success` is internally adjusted
            to be capped by the number of realizations. It is possible to set
            the minimum number of successful realizations equal to zero. Some
            optimization algorithms are able to handle this and will proceed
            even if all realizations failed. Most algorithms are not capable of
            this and will internally adjust the value to be equal to one.
            """
        ),
    )
    perturbation_num: int | None = Field(
        default=None,
        gt=0,
        description=dedent(
            """
            The number of perturbed control vectors per realization.

            The number of simulation runs used for estimating the gradient is
            equal to the the product of `perturbation_num` and
            `model.realizations`.
            """
        ),
    )
    speculative: bool = Field(
        default=False,
        description=dedent(
            """
            Calculate gradients, even if not strictly needed.

            When running forward models in parallel, e.g. on a computing
            cluster, it may be advantageous to calculate a gradient along with
            each function evaluation. The reason is that the optimizer may be
            able to use the gradient in a next iteration. Although this wastes
            resources if the gradient is not used, it may save clock time, since
            the gradient does not need to be calculated again.

            Notes:

            - This is mostly useful if a full set of functions and perturbations
              (for the gradient) can run in parallel. Additional batches may be
              needed anyway if this is not the case.
            - Running an optimization with or without this disabled will yield
              slightly different results, since the perturbed variables will
              differ because different (longer) sequences of random numbers will
              be generated.
            - This does potentially waste a lot of computational resources!
            """
        ),
    )
    parallel: bool = Field(
        default=True,
        description=dedent(
            """
            Enable parallel function evaluation.

            If set to `False`, a single function evaluation is performed in each
            batch. In case of gradient-free optimizers this can be highly
            inefficient, since these tend to need many independent function
            evaluations at each iteration. By setting parallel to `True`,
            multiple functions may be evaluated in parallel, if supported by the
            optimization algorithm.

            The default is to use parallel evaluation if supported.
            """
        ),
    )
    auto_scale: bool = Field(
        default=False,
        description=dedent(
            """
            Enable auto-scaling of objectives and input/output constraints.

            If set to `True`, objectives, input constraints and output
            constraints are automatically scaled.

            In the case of objectives and output constraints this is done by
            scaling by the values obtained in the first batch. Input constraints
            are scaled by dividing by the maximum values of their coefficients
            or right-hand-sides.

            Note: If enabled, the individual scale options for objectives and
            constraints are not allowed.
            """
        ),
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

        try:
            plugin_name = find_optimizer_plugin(algorithm)
        except ValueError:
            raise
        except Exception as exc:
            ert_version = importlib.metadata.version("ert")
            ropt_version = importlib.metadata.version("ropt")
            msg = (
                f"Error while initializing ropt:\n\n{exc}.\n\n"
                "There may a be version mismatch between "
                f"ERT ({ert_version}) and ropt ({ropt_version})\n"
                "If the installation is correct, please report this as a bug."
            )
            raise RuntimeError(msg) from exc
        if plugin_name is None:
            raise ValueError(f"Optimizer algorithm '{algorithm}' not found")
        self._optimization_plugin_name = plugin_name

        options = self.options or self.backend_options
        if options:
            validate_optimizer_options(algorithm, options)

        self.backend = None
        self.algorithm = algorithm

        return self

    @property
    def optimization_plugin_name(self) -> str:
        return self._optimization_plugin_name
