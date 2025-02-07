from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from ropt.transforms import OptModelTransforms
from ropt.transforms.base import ObjectiveTransform, VariableTransform

from everest.config import ControlConfig, ObjectiveFunctionConfig
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

    def forward(self, objectives: NDArray[np.float64]) -> NDArray[np.float64]:
        return objectives / self._scales

    def backward(self, objectives: NDArray[np.float64]) -> NDArray[np.float64]:
        return objectives * self._scales

    def calculate_auto_scales(self, objectives: NDArray[np.float64]) -> None:
        auto_scales = np.abs(
            np.nansum(objectives * self._weights[:, np.newaxis], axis=0)
        )
        self._scales[self._auto_scales] *= auto_scales[self._auto_scales]

    @property
    def has_auto_scale(self) -> bool:
        return bool(np.any(self._auto_scales))


def get_optimization_domain_transforms(
    controls: list[ControlConfig],
    objectives: list[ObjectiveFunctionConfig],
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
    )
