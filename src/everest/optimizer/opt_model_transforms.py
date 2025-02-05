from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from ropt.transforms import OptModelTransforms
from ropt.transforms.base import (
    NonLinearConstraintTransform,
    ObjectiveTransform,
    VariableTransform,
)

from everest.config import (
    ControlConfig,
    ObjectiveFunctionConfig,
    OutputConstraintConfig,
)
from everest.config.utils import FlattenedControls


class ControlScaler(VariableTransform):
    def __init__(
        self,
        lower_bounds: Sequence[float],
        upper_bounds: Sequence[float],
        scaled_ranges: Sequence[tuple[float, float]],
        auto_scales: Sequence[bool],
    ) -> None:
        self._scales = [
            (ub - lb) / (sr[1] - sr[0]) if au else 1.0
            for au, lb, ub, sr in zip(
                auto_scales, lower_bounds, upper_bounds, scaled_ranges, strict=True
            )
        ]
        self._offsets = [
            lb - sr[0] * sc if au else 0.0
            for au, lb, sc, sr in zip(
                auto_scales, lower_bounds, self._scales, scaled_ranges, strict=True
            )
        ]

    def forward(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        return (values - self._offsets) / self._scales

    def backward(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        return values * self._scales + self._offsets

    def transform_magnitudes(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        return values / self._scales

    def transform_linear_constraints(  # type: ignore
        self,
        coefficients: NDArray[np.float64],
        lower_bounds: NDArray[np.float64],
        upper_bounds: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        r"""Transform a set of linear constraints.

        The set of linear constraints can be represented by a matrix equation:
        $\mathbf{A} \mathbf{x} = \mathbf{b}.

        When rescaling variables, the linear coefficients ($\mathbf{A}$) and
        right-hand-side values ($\mathbf{b}$) must be converted to remain
        valid for the scaled variables:
        $$
        \begin{align}
            \hat{\mathbf{A}} &= \mathbf{A} \mathbf{S} \\
            \hat{\mathbf{b}} &= \mathbf{b} - \mathbf{A}\mathbf{o}
        \end{align}
        $$

        where $\mathbf{S}$ is a diagonal matrix containing the variable
        scales, and $\mathbf{o}$ is a vector containing the variable offsets.
        """
        if self._offsets is not None:
            offsets = np.matmul(coefficients, self._offsets)
            lower_bounds = lower_bounds - offsets  # noqa: PLR6104
            upper_bounds = upper_bounds - offsets  # noqa: PLR6104
        if self._scales is not None:
            coefficients = coefficients * self._scales  # noqa: PLR6104
        return coefficients, lower_bounds, upper_bounds


class ObjectiveScaler(ObjectiveTransform):
    def __init__(
        self, scales: list[float], auto_scales: list[bool], weights: list[float]
    ) -> None:
        self._scales = np.asarray(scales, dtype=np.float64)
        self._auto_scales = np.asarray(auto_scales, dtype=np.bool_)
        self._weights = np.asarray(weights, dtype=np.float64)

    # The transform methods below all return the negative of the objectives.
    # This is because Everest maximizes the objectives, while ropt is a minimizer.

    def forward(self, objectives: NDArray[np.float64]) -> NDArray[np.float64]:
        return -objectives / self._scales

    def backward(self, objectives: NDArray[np.float64]) -> NDArray[np.float64]:
        return -objectives * self._scales

    def transform_weighted_objective(self, weighted_objective):  # type: ignore
        return -weighted_objective

    def calculate_auto_scales(self, objectives: NDArray[np.float64]) -> None:
        auto_scales = np.abs(
            np.nansum(objectives * self._weights[:, np.newaxis], axis=0)
        )
        self._scales[self._auto_scales] *= auto_scales[self._auto_scales]

    @property
    def has_auto_scale(self) -> bool:
        return bool(np.any(self._auto_scales))


class ConstraintScaler(NonLinearConstraintTransform):
    def __init__(
        self, scales: list[float], auto_scales: list[bool], weights: list[float]
    ) -> None:
        self._scales = np.asarray(scales, dtype=np.float64)
        self._auto_scales = np.asarray(auto_scales, dtype=np.bool_)
        self._weights = np.asarray(weights, dtype=np.float64)

    def transform_bounds(
        self, lower_bounds: NDArray[np.float64], upper_bounds: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return lower_bounds / self._scales, upper_bounds / self._scales

    def forward(self, constraints: NDArray[np.float64]) -> NDArray[np.float64]:
        return constraints / self._scales

    def backward(self, constraints: NDArray[np.float64]) -> NDArray[np.float64]:
        return constraints * self._scales

    def calculate_auto_scales(self, constraints: NDArray[np.float64]) -> None:
        auto_scales = np.abs(
            np.nansum(constraints * self._weights[:, np.newaxis], axis=0)
        )
        self._scales[self._auto_scales] *= auto_scales[self._auto_scales]

    @property
    def has_auto_scale(self) -> bool:
        return bool(np.any(self._auto_scales))


def get_optimization_domain_transforms(
    controls: list[ControlConfig],
    objectives: list[ObjectiveFunctionConfig],
    constraints: list[OutputConstraintConfig] | None,
    weights: list[float],
) -> OptModelTransforms:
    flattened_controls = FlattenedControls(controls)
    return OptModelTransforms(
        variables=(
            ControlScaler(
                flattened_controls.lower_bounds,
                flattened_controls.upper_bounds,
                flattened_controls.scaled_ranges,
                flattened_controls.auto_scales,
            )
            if any(flattened_controls.auto_scales)
            else None
        ),
        objectives=ObjectiveScaler(
            [
                1.0 if objective.scale is None else objective.scale
                for objective in objectives
            ],
            [
                False if objective.auto_scale is None else objective.auto_scale
                for objective in objectives
            ],
            weights,
        ),
        nonlinear_constraints=(
            ConstraintScaler(
                [
                    1.0 if constraint.scale is None else constraint.scale
                    for constraint in constraints
                ],
                [
                    False if constraint.auto_scale is None else constraint.auto_scale
                    for constraint in constraints
                ],
                weights,
            )
            if constraints
            else None
        ),
    )
