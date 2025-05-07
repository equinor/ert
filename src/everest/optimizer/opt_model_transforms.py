from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypedDict

import numpy as np
from numpy.typing import NDArray
from ropt.transforms.base import (
    NonLinearConstraintTransform,
    ObjectiveTransform,
    VariableTransform,
)

from everest.config import (
    ControlConfig,
    ModelConfig,
    ObjectiveFunctionConfig,
    OutputConstraintConfig,
)
from everest.config.utils import FlattenedControls


class EverestOptModelTransforms(TypedDict):
    control_scaler: ControlScaler
    objective_scaler: ObjectiveScaler
    constraint_scaler: ConstraintScaler | None


class ControlScaler(VariableTransform):
    """Transformation object to define a linear scaling of controls.

    This object defines a linear scaling from lower and upper bounds [lb, ub] in
    the user domain to a target range in the optimizer domain.
    """

    def __init__(
        self,
        lower_bounds: Sequence[float],
        upper_bounds: Sequence[float],
        scaled_ranges: Sequence[tuple[float, float]],
        control_types: list[Literal["real", "integer"]],
    ) -> None:
        """Transformation object to define a linear scaling.

        This is implemented by internally representing the transformation from
        the user to the optimizer domain by a subtraction of an offset and a
        division by a scaling factor.

         Args:
             lower_bounds:  Lower bounds in the user domain.
             upper_bounds:  Upper bounds in the user domain.
             scaled_ranges: target ranges in the optimizer domain.
             control_types: Types of the controls, real or integer.
        """
        self._scaling_factors = [
            (ub - lb) / (sr[1] - sr[0]) if ct == "real" else 1.0
            for lb, ub, sr, ct in zip(
                lower_bounds, upper_bounds, scaled_ranges, control_types, strict=True
            )
        ]
        self._offsets = [
            lb - sr[0] * sc if ct == "real" else 0.0
            for lb, sc, sr, ct in zip(
                lower_bounds,
                self._scaling_factors,
                scaled_ranges,
                control_types,
                strict=True,
            )
        ]

    def to_optimizer(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform values to the optimizer domain.

        The transformation is defined by subtracting offsets, followed by
        division by scaling factors.

        Args:
            values: The values to transform

        Returns:
            The transformed values.
        """
        return (values - self._offsets) / self._scaling_factors

    def from_optimizer(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        return values * self._scaling_factors + self._offsets

    def magnitudes_to_optimizer(
        self, values: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Transform a magnitude value to the optimizer domain.

        Since magnitudes are relative values only a scaling is applied.

        Args:
            values: The magnitudes to transform.

        Returns:
            The transformed magnitudes.
        """
        return values / self._scaling_factors

    def linear_constraints_to_optimizer(
        self,
        coefficients: NDArray[np.float64],
        lower_bounds: NDArray[np.float64],
        upper_bounds: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        r"""Transform a set of linear constraints.

        The set of linear constraints can be represented by a matrix equation:
        $\mathbf{A} \mathbf{x} = \mathbf{b}$. When linearly transforming
        variables to the optimizer domain, the coefficients ($\mathbf{A}$) and
        right-hand-side values ($\mathbf{b}$) must be converted to remain valid.

        Here, the linear transformation of the variables to the optimizer domain
        is given by the scaling factors $\mathbf{s}_i$ and the offsets $\mathbf{o}_i$:

        $$
        \hat{\mathbf{x}}_i = \frac{\mathbf{x}_i - \mathbf{o}_i}{\mathbf{s}_i}
        $$

        In the optimizer domeain, the coefficients and right-hand-side values
        must then be transformed as follows:

        $$ \begin{align}
            \hat{\mathbf{A}} &= \mathbf{A} \mathbf{S} \\
            \hat{\mathbf{b}} &= \mathbf{b} - \mathbf{A}\mathbf{o}
        \end{align}$$

        where $\mathbf{S}$ is a diagonal matrix containing the variable scales
        $\mathbf{s}_i$.
        """
        if self._offsets is not None:
            offsets = np.matmul(coefficients, self._offsets)
            lower_bounds = lower_bounds - offsets  # noqa: PLR6104
            upper_bounds = upper_bounds - offsets  # noqa: PLR6104
        if self._scaling_factors is not None:
            coefficients = coefficients * self._scaling_factors  # noqa: PLR6104
        return coefficients, lower_bounds, upper_bounds

    def bound_constraint_diffs_from_optimizer(
        self, lower_diffs: NDArray[np.float64], upper_diffs: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Transform constraint differences to the user domain.

        Since these values are differences with respect to a bound, they are not
        affected by offsets. They are transformed back by multiplying with the
        scaling factor.

        Args:
            lower_diffs: Differences with respect to the lower bounds.
            upper_diffs: Differences with respect to the upper bounds.

        Returns:
            The re-scaled bounds.
        """
        if self._scaling_factors is not None:
            lower_diffs = lower_diffs * self._scaling_factors  # noqa: PLR6104
            upper_diffs = upper_diffs * self._scaling_factors  # noqa: PLR6104
        return lower_diffs, upper_diffs

    def linear_constraints_diffs_from_optimizer(
        self, lower_diffs: NDArray[np.float64], upper_diffs: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Transform linear constraint differences to the user domain.

        Linear constraints are transformed to remain valid in the optimizer
        domain, but the equations themselves are not scaled. Hence differences
        with the right-hand-side are the same in the optimization and user
        domain.

        Args:
            lower_diffs: Differences with respect to the lower bounds.
            upper_diffs: Differences with respect to the upper bounds.

        Note:
            This method is defined in the base class to raise a not-implemented
            error, hence it must be overridden even if it just returns its
            inputs. The transformation logic in `ropt` expects valid results to
            be returned.

        Returns:
            The original inputs.
        """
        return lower_diffs, upper_diffs


class ObjectiveScaler(ObjectiveTransform):
    """Transformation object for linearly scaling objectives.

    Objectives are transformed to the optimizer domain by division with a
    scaling factor. Some scaling factors may be calculated after initialization
    of the object. The `has_auto_scales` property will be true if this
    calculation is needed, and the `calculate_auto_scales` method can be used to
    calculate this for each objective from a set of realizations of the objective.

    In addition to scaling, this object also implements the transformation from
    maximization to minimization by multiplying the objectives with -1.
    """

    def __init__(
        self, scales: list[float], auto_scales: list[bool], weights: list[float]
    ) -> None:
        """Initialize the object.

        Args:
            scales:      The scales.
            auto_scales: Flags indicating which objects are auto-scaled.
            weights:     Weights used to perform auto-scaling.
        """
        self._scales = np.asarray(scales, dtype=np.float64)
        self._auto_scales = np.asarray(auto_scales, dtype=np.bool_)
        self._weights = np.asarray(weights, dtype=np.float64)
        self._needs_auto_scale_calculation = bool(np.any(self._auto_scales))

    # The transform methods below all return the negative of the objectives.
    # This is because Everest maximizes the objectives, while ropt is a minimizer.

    def to_optimizer(self, objectives: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform objectives to optimizer space.

        Args:
            objectives: The objectives to transform.

        Returns:
            The negative of the scaled objectives.
        """
        return -objectives / self._scales

    def from_optimizer(self, objectives: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform objectives to user space.

        Args:
            objectives: The objectives to transform.

        Returns:
            The negative of the re-scaled objectives.
        """
        return -objectives * self._scales

    def weighted_objective_from_optimizer(
        self, weighted_objective: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Transform the weighted objective to user space.

        The weighted objective is composed of several objectives that may be
        scaled different. By definition scaling it back somehow is meaningless.
        However, due to the transformation from maximization to minimization, a
        factor of -1 is applied.

        Args:
            weighted_objective: The weighted objective value.

        Returns:
            The value multiplied with -1.
        """
        return -weighted_objective

    def calculate_auto_scales(
        self, objectives: NDArray[np.float64], realizations: NDArray[np.intc]
    ) -> None:
        """Calculated scales from a set of realizations.

        For selected objectives, this method calculates the scales by a weighted
        sum of the objectives from an ensemble of objective values.

        The `objectives` argument must be a matrix, where each row represents a
        different realization of each objective. The indices of the realizations
        must be passed via the `realizations` argument.

        Args:
            objectives:   The objectives for each realization.
            realizations: The realizations to use for the calculation.
        """
        weights = np.tile(self._weights[realizations, np.newaxis], objectives.shape[1])
        weights[np.isnan(objectives)] = 0.0
        weights /= np.sum(weights, axis=0)
        auto_scales = np.abs(np.sum(objectives * weights, axis=0))
        auto_scales = np.where(auto_scales > 0.0, auto_scales, 1.0)
        self._scales[self._auto_scales] *= auto_scales[self._auto_scales]
        self._needs_auto_scale_calculation = False

    @property
    def needs_auto_scale_calculation(self) -> bool:
        """Return true auto-scaling must be initialized"""
        return self._needs_auto_scale_calculation


class ConstraintScaler(NonLinearConstraintTransform):
    """Transformation object for linearly scaling constraints.

    Constraints are transformed to the optimizer domain by division with a
    scaling factor. Some scaling factors may be calculated after initialization
    of the object. The `has_auto_scales` property will be true if this
    calculation is needed, and the `calculate_auto_scales` method can be used to
    calculate this for each constraint from a set of realizations of the constraint.
    """

    def __init__(
        self, scales: list[float], auto_scales: list[bool], weights: list[float]
    ) -> None:
        """Initialize the object.

        Args:
            scales:      The scales.
            auto_scales: Flags indicating which constraints are auto-scaled.
            weights:     Weights used to perform auto-scaling.
        """
        self._scales = np.asarray(scales, dtype=np.float64)
        self._auto_scales = np.asarray(auto_scales, dtype=np.bool_)
        self._weights = np.asarray(weights, dtype=np.float64)
        self._needs_auto_scale_calculation = bool(np.any(self._auto_scales))

    def bounds_to_optimizer(
        self, lower_bounds: NDArray[np.float64], upper_bounds: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Transform the bounds of teh constraints to the optimizer domain.

        Args:
             lower_bounds:  Lower bounds in the user domain.
             upper_bounds:  Upper bounds in the user domain.

        Returns:
            The scaled bounds.
        """
        return lower_bounds / self._scales, upper_bounds / self._scales

    def to_optimizer(self, constraints: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform constraints to optimizer space.

        Args:
            constraints: The constraints to transform.

        Returns:
            The scaled constraints.
        """
        return constraints / self._scales

    def from_optimizer(self, constraints: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform constraints to user space.

        Args:
            constraints: The constraints to transform.

        Returns:
            The negative of the re-scaled constraints.
        """
        return constraints * self._scales

    def nonlinear_constraint_diffs_from_optimizer(
        self, lower_diffs: NDArray[np.float64], upper_diffs: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Transform constraint differences to the user domain.

        Args:
            lower_diffs: Differences with respect to the lower bounds.
            upper_diffs: Differences with respect to the upper bounds.

        Returns:
            The re-scaled bounds.
        """
        return lower_diffs * self._scales, upper_diffs * self._scales

    def calculate_auto_scales(
        self, constraints: NDArray[np.float64], realizations: NDArray[np.intc]
    ) -> None:
        """Calculated scales from a set of realizations.

        For selected constraints, this method calculates the scales by a
        weighted sum of the constraints from an ensemble of constraint values.

        The `constraints` argument must be a matrix, where each row represents a
        different realization of each constraint. The indices of the
        realizations must be passed via the `realizations` argument.

        Args:
            constraints:  The constraints for each realization.
            realizations: The realizations to use for the calculation.
        """
        weights = np.tile(self._weights[realizations, np.newaxis], constraints.shape[1])
        weights[np.isnan(constraints)] = 0.0
        weights /= np.sum(weights, axis=0)
        auto_scales = np.abs(np.sum(constraints * weights, axis=0))
        auto_scales = np.where(auto_scales > 0.0, auto_scales, 1.0)
        self._scales[self._auto_scales] *= auto_scales[self._auto_scales]
        self._needs_auto_scale_calculation = False

    @property
    def needs_auto_scale_calculation(self) -> bool:
        """Return true if auto-scaling must be initialized"""
        return self._needs_auto_scale_calculation


def get_optimization_domain_transforms(
    controls: list[ControlConfig],
    objectives: list[ObjectiveFunctionConfig],
    constraints: list[OutputConstraintConfig] | None,
    model: ModelConfig,
) -> EverestOptModelTransforms:
    flattened_controls = FlattenedControls(controls)
    control_scaler = ControlScaler(
        flattened_controls.lower_bounds,
        flattened_controls.upper_bounds,
        flattened_controls.scaled_ranges,
        flattened_controls.types,
    )

    weights = (
        [1.0 / len(model.realizations)] * len(model.realizations)
        if model.realizations_weights is None
        else model.realizations_weights
    )

    objective_scaler = ObjectiveScaler(
        [
            1.0 if objective.scale is None else objective.scale
            for objective in objectives
        ],
        [
            False if objective.auto_scale is None else objective.auto_scale
            for objective in objectives
        ],
        weights,
    )

    constraint_scaler = (
        ConstraintScaler(
            [
                1.0 if constraint.scale is None else constraint.scale
                for constraint in constraints
            ],
            [constraint.auto_scale for constraint in constraints],
            weights,
        )
        if constraints
        else None
    )

    return {
        "control_scaler": control_scaler,
        "objective_scaler": objective_scaler,
        "constraint_scaler": constraint_scaler,
    }
