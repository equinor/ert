from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from ropt.enums import ConstraintType
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

    def transform_linear_constraints(
        self, coefficients: NDArray[np.float64], rhs_values: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return (
            coefficients * self._scales,
            rhs_values - np.matmul(coefficients, self._offsets),
        )


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

    def transform_weighted_objective(self, weighted_objective):
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

    def transform_rhs_values(
        self, rhs_values: NDArray[np.float64], types: NDArray[np.ubyte]
    ) -> tuple[NDArray[np.float64], NDArray[np.ubyte]]:
        def flip_type(constraint_type: ConstraintType) -> ConstraintType:
            match constraint_type:
                case ConstraintType.GE:
                    return ConstraintType.LE
                case ConstraintType.LE:
                    return ConstraintType.GE
                case _:
                    return constraint_type

        rhs_values = rhs_values / self._scales  # noqa: PLR6104
        # Flip inequality types if self._scales < 0 in the division above:
        types = np.fromiter(
            (
                flip_type(type_) if scale < 0 else type_
                for type_, scale in zip(types, self._scales, strict=False)
            ),
            np.ubyte,
        )
        return rhs_values, types

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
