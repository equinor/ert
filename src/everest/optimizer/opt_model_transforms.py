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

from ert.config import EverestConstraintsConfig, EverestControl, EverestObjectivesConfig
from everest.config import (
    InputConstraintConfig,
    ModelConfig,
)


class EverestOptModelTransforms(TypedDict):
    control_scaler: ControlScaler
    objective_scaler: ObjectiveScaler
    constraint_scaler: ConstraintScaler | None


class ControlScaler(VariableTransform):
    """Transformation object to define scaling related to the scales.

    For scaling of the controls itself, this object defines a linear scaling
    from lower and upper bounds [lb, ub] in the user domain to a target range in
    the optimizer domain.

    Constraints on linear combinations on the controls are defined by the
    `input_constraints` section of the configuration. If a linear transformation
    of the controls is defined, the linear
     constraints are also transformed accordingly. In
    addition, each of the linear constraints can be scaled by an overall factor,
    either automatically if the `auto_scale` option in the `optimization`
    section is set, or manually if the `scale` option in the `input_constraints`
    section is set.
    """

    def __init__(
        self,
        lower_bounds: Sequence[float],
        upper_bounds: Sequence[float],
        scaled_ranges: Sequence[tuple[float, float]],
        control_types: list[Literal["real", "integer"]],
        auto_scale_input_constraints: bool,
        input_constraint_scales: list[float] | None,
    ) -> None:
        """Transformation object to define a linear scaling.

        This is implemented by internally representing the transformation from
        the user to the optimizer domain by a subtraction of an offset and a
        division by a scaling factor.

         Args:
             lower_bounds:                 Lower bounds in the user domain.
             upper_bounds:                 Upper bounds in the user domain.
             scaled_ranges:                Target ranges in the optimizer domain.
             control_types:                Types of the controls, real or integer.
             auto_scale_input_constraints: Auto-scale any input constraint equations.
             input_constraint_scales:      Optional scaling factors of input constraints
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
        self._auto_scale_input_constraints = auto_scale_input_constraints
        self._input_constraint_scales = input_constraint_scales

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

        The transformation consists of two steps:

        1. Transformation to correct for variable scaling:

           The set of linear constraints can be represented by a matrix
           equation: Ax = b. When linearly transforming variables to the
           optimizer domain, the coefficients A and right-hand-side values b
           must be converted to remain valid.

           the linear transformation of the variables to the optimizer domain is
           given by the scaling factors s and the offsets o:

               x = (x - o) / s.

           In the optimizer domain, the coefficients and right-hand-side values
           must then be transformed as follows:

               A = AS
               b = b - Ao

           where S is a diagonal matrix containing the variable scales s.

        2. Transformation to correct for constraint scaling:

           Each linear equation is scaled by a constant value that is either
           determined automatically or manually set.
        """
        # The inputs may be immutable arrays, hence the `noqa PLR6104`

        # Correct for variable scaling:
        offsets = np.matmul(coefficients, self._offsets)
        lower_bounds = lower_bounds - offsets  # noqa: PLR6104
        upper_bounds = upper_bounds - offsets  # noqa: PLR6104
        coefficients = coefficients * self._scaling_factors  # noqa: PLR6104

        # Correct for constraint scaling:
        if self._auto_scale_input_constraints:
            scales = np.max(
                [
                    np.where(np.isfinite(lower_bounds), np.abs(lower_bounds), 0.0),
                    np.where(np.isfinite(upper_bounds), np.abs(upper_bounds), 0.0),
                    np.max(np.abs(coefficients), axis=1),
                ],
                axis=0,
            )
        else:
            assert self._input_constraint_scales is not None
            scales = np.asarray(self._input_constraint_scales, np.float64)
        coefficients = coefficients / scales[:, np.newaxis]  # noqa: PLR6104
        lower_bounds = lower_bounds / scales  # noqa: PLR6104
        upper_bounds = upper_bounds / scales  # noqa: PLR6104

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
            # The inputs may be immutable arrays, hence the `noqa PLR6104`
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

        Returns:
            The original inputs.
        """
        assert self._input_constraint_scales is not None
        # The inputs may be immutable arrays, hence the `noqa PLR6104`
        lower_diffs = lower_diffs * self._input_constraint_scales  # noqa: PLR6104
        upper_diffs = upper_diffs * self._input_constraint_scales  # noqa: PLR6104
        return lower_diffs, upper_diffs


class ObjectiveScaler(ObjectiveTransform):
    """Transformation object for linearly scaling objectives.

    Objectives are transformed to the optimizer domain by division with a
    scaling factor. The scaling factors may be calculated after initialization
    of the object. The `auto_scale` property will be true if this calculation is
    needed, and the `calculate_auto_scales` method can be used to calculate this
    for each objective from a set of realizations of the objective.

    In addition to scaling, this object also implements the transformation from
    maximization to minimization by multiplying the objectives with -1.
    """

    def __init__(
        self,
        auto_scale: bool,
        scales: list[float],
        realization_weights: list[float],
        objective_weights: list[float],
    ) -> None:
        """Initialize the object.

        Args:
            auto_scale:          Flag indicating if the objects are auto-scaled.
            scales:              The scales of the objectives, 1 if auto-scaling.
            realization_weights: The weights of the realizations.
        """
        self._auto_scale = auto_scale
        self._scales = np.asarray(scales, dtype=np.float64)
        self._realization_weights = np.asarray(realization_weights, dtype=np.float64)
        self._objective_weights = np.asarray(objective_weights, dtype=np.float64)
        self._objective_weights /= np.sum(self._objective_weights)

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
        if self._auto_scale:
            error_msg = (
                "Auto-scaling of the objective failed "
                "to estimate a positive scale factor"
            )
            avg_objectives = _avg_functions(
                realizations, self._realization_weights, objectives, error_msg
            )
            self._scales = np.abs(np.dot(avg_objectives, self._objective_weights))
            if self._scales < np.finfo(np.float64).eps:
                raise RuntimeError(error_msg)
        self._auto_scale = False

    @property
    def needs_auto_scale_calculation(self) -> bool:
        """Return true auto-scaling must be initialized"""
        return self._auto_scale


class ConstraintScaler(NonLinearConstraintTransform):
    """Transformation object for linearly scaling constraints.

    Constraints are transformed to the optimizer domain by division with a
    scaling factor. The scaling factors may be calculated after initialization
    of the object. The `auto_scale` property will be true if this calculation is
    needed, and the `calculate_auto_scales` method can be used to calculate this
    for each constraint from a set of realizations of the constraint.
    """

    def __init__(
        self, auto_scale: bool, scales: list[float], realization_weights: list[float]
    ) -> None:
        """Initialize the object.

        Args:
            auto_scale:          Flag indicating if constraints are auto-scaled.
            scales:              The scales of the constraints, 1 if auto-scaling.
            realization_weights: The weights of the realizations.
        """
        self._auto_scale = auto_scale
        self._scales = np.asarray(scales, dtype=np.float64)
        self._realization_weights = np.asarray(realization_weights, dtype=np.float64)

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
        if self._auto_scale:
            error_msg = (
                "Auto-scaling of the constraints failed "
                "to estimate a positive scale factor"
            )
            avg_constraints = _avg_functions(
                realizations, self._realization_weights, constraints, error_msg
            )
            self._scales = np.abs(avg_constraints)
            if np.any(self._scales < np.finfo(np.float64).eps):
                raise RuntimeError(error_msg)
        self._auto_scale = False

    @property
    def needs_auto_scale_calculation(self) -> bool:
        """Return true if auto-scaling must be initialized"""
        return self._auto_scale


def get_optimization_domain_transforms(
    controls: list[EverestControl],
    objectives: EverestObjectivesConfig,
    input_constraints: list[InputConstraintConfig] | None,
    output_constraints: EverestConstraintsConfig | None,
    model: ModelConfig,
    auto_scale: bool,
) -> EverestOptModelTransforms:
    control_scaler = ControlScaler(
        [control.min for control in controls],
        [control.max for control in controls],
        [control.scaled_range for control in controls],
        [control.variable_type for control in controls],
        auto_scale_input_constraints=auto_scale,
        input_constraint_scales=(
            None
            if input_constraints is None
            else [
                1.0 if constraint.scale is None else constraint.scale
                for constraint in input_constraints
            ]
        ),
    )

    realization_weights = (
        [1.0 / len(model.realizations)] * len(model.realizations)
        if model.realizations_weights is None
        else model.realizations_weights
    )

    objective_scaler = ObjectiveScaler(
        auto_scale=auto_scale,
        scales=objectives.scales,
        realization_weights=realization_weights,
        objective_weights=objectives.weights,
    )

    constraint_scaler = (
        ConstraintScaler(
            auto_scale=auto_scale,
            scales=output_constraints.scales,
            realization_weights=realization_weights,
        )
        if output_constraints
        else None
    )

    return {
        "control_scaler": control_scaler,
        "objective_scaler": objective_scaler,
        "constraint_scaler": constraint_scaler,
    }


def _avg_functions(
    realizations: NDArray[np.intc],
    realization_weights: NDArray[np.float64],
    functions: NDArray[np.float64],
    error_msg: str,
) -> NDArray[np.float64]:
    realization_weights = np.tile(
        realization_weights[realizations, np.newaxis], functions.shape[1]
    )
    realization_weights[np.isnan(functions)] = 0.0
    rw_sum = np.sum(realization_weights, axis=0)
    if np.any(rw_sum < np.finfo(np.float64).eps):
        raise RuntimeError(error_msg)
    realization_weights /= rw_sum
    return np.sum(functions * realization_weights, axis=0)
