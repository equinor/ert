from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from ropt.transforms.base import VariableTransform

from ert.config import EverestControl
from everest.config import InputConstraintConfig


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
        self._scaling_factors = np.asarray(
            [
                (ub - lb) / (sr[1] - sr[0]) if ct == "real" else 1.0
                for lb, ub, sr, ct in zip(
                    lower_bounds,
                    upper_bounds,
                    scaled_ranges,
                    control_types,
                    strict=True,
                )
            ],
            dtype=np.float64,
        )
        self._offsets = np.asarray(
            [
                lb - sr[0] * sc if ct == "real" else 0.0
                for lb, sc, sr, ct in zip(
                    lower_bounds,
                    self._scaling_factors,
                    scaled_ranges,
                    control_types,
                    strict=True,
                )
            ],
            dtype=np.float64,
        )
        self._auto_scale_input_constraints = auto_scale_input_constraints
        self._input_constraint_scales = input_constraint_scales

    def set_free_mask(self, mask: NDArray[np.bool_]) -> None:
        """Restrict the transformation to the free controls.

        The scaling factors and offsets of the fixed controls are set to 1 and
        0, respectively, so that those controls are not transformed.

        Args:
            mask: A boolean array indicating which controls are free.
        """
        self._scaling_factors = np.where(mask, self._scaling_factors, 1.0)
        self._offsets = np.where(mask, self._offsets, 0.0)

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
        lower_bounds = lower_bounds - offsets  # ruff: ignore[non-augmented-assignment]
        upper_bounds = upper_bounds - offsets  # ruff: ignore[non-augmented-assignment]
        coefficients = coefficients * self._scaling_factors  # ruff: ignore[non-augmented-assignment]

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
        coefficients = coefficients / scales[:, np.newaxis]  # ruff: ignore[non-augmented-assignment]
        lower_bounds = lower_bounds / scales  # ruff: ignore[non-augmented-assignment]
        upper_bounds = upper_bounds / scales  # ruff: ignore[non-augmented-assignment]

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
            lower_diffs = lower_diffs * self._scaling_factors  # ruff: ignore[non-augmented-assignment]
            upper_diffs = upper_diffs * self._scaling_factors  # ruff: ignore[non-augmented-assignment]
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
        lower_diffs = lower_diffs * self._input_constraint_scales  # ruff: ignore[non-augmented-assignment]
        upper_diffs = upper_diffs * self._input_constraint_scales  # ruff: ignore[non-augmented-assignment]
        return lower_diffs, upper_diffs


def get_control_scaler(
    controls: list[EverestControl],
    input_constraints: list[InputConstraintConfig] | None,
    auto_scale: bool,
) -> ControlScaler:
    return ControlScaler(
        [control.min for control in controls],
        [control.max for control in controls],
        [control.scaled_range for control in controls],
        [control.control_type for control in controls],
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
