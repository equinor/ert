from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from ropt.transforms import OptModelTransforms
from ropt.transforms.base import VariableTransform

from everest.config import ControlConfig
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


def get_opt_model_transforms(controls: list[ControlConfig]) -> OptModelTransforms:
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
        )
    )
