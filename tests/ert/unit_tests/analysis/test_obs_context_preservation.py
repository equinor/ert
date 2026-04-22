from __future__ import annotations

import numpy as np
import pytest

from ert.analysis._update_strategies._adaptive import AdaptiveLocalizationUpdate
from ert.analysis._update_strategies._distance import DistanceLocalizationUpdate
from ert.analysis._update_strategies._protocol import (
    ObservationContext,
    ObservationLocations,
)
from ert.analysis._update_strategies._standard import StandardESUpdate
from ert.config import Field


def _noop(_event: object) -> None:
    pass


def _make_context(
    *, n_obs: int = 6, n_real: int = 20, obs_mean: float = 100.0
) -> ObservationContext:
    """Build an ObservationContext with response row means far from zero so
    that any in-place mean-subtraction is clearly visible.
    """
    rng = np.random.default_rng(0)
    responses = (rng.standard_normal((n_obs, n_real)) + obs_mean).astype(np.float64)
    obs_values = np.full(n_obs, obs_mean)
    obs_errors = np.ones(n_obs)
    # location info for every obs so distance localization is usable
    obs_loc = ObservationLocations(
        xpos=np.linspace(0.5, n_obs - 0.5, n_obs),
        ypos=np.full(n_obs, 0.5),
        main_range=np.full(n_obs, 5.0),
        location_mask=np.ones(n_obs, dtype=bool),
    )
    return ObservationContext(
        responses=responses,
        observation_values=obs_values,
        observation_errors=obs_errors,
        observation_locations=obs_loc,
    )


def test_that_standard_es_prepare_does_not_mutate_obs_context_responses() -> None:
    ctx = _make_context()
    original = ctx.responses.copy()

    strat = StandardESUpdate(
        enkf_truncation=1.0,
        rng=np.random.default_rng(0),
        progress_callback=_noop,
    )
    strat.prepare(ctx)

    np.testing.assert_array_equal(ctx.responses, original)


def test_that_adaptive_localization_prepare_does_not_mutate_obs_context_responses() -> (
    None
):
    ctx = _make_context()
    original = ctx.responses.copy()

    strat = AdaptiveLocalizationUpdate(
        correlation_threshold=lambda _n: 0.0,
        enkf_truncation=1.0,
        rng=np.random.default_rng(0),
        progress_callback=_noop,
    )
    strat.prepare(ctx)

    np.testing.assert_array_equal(ctx.responses, original)


def test_that_distance_localization_prepare_does_not_mutate_obs_context_responses() -> (
    None
):
    ctx = _make_context()
    original = ctx.responses.copy()

    strat = DistanceLocalizationUpdate(
        enkf_truncation=1.0,
        rng=np.random.default_rng(0),
        param_type=Field,
        progress_callback=_noop,
    )
    strat.prepare(ctx)

    np.testing.assert_array_equal(ctx.responses, original)


def test_that_distance_localization_innovation_is_independent_of_prepare_order() -> (
    None
):
    """
    The distance smoother's internal D_obs_minus_D must not depend on whether
    StandardESUpdate.prepare ran before or after DistanceLocalizationUpdate.prepare.
    """

    def _prepare_pair(standard_first: bool) -> np.ndarray:
        ctx = _make_context()
        dist = DistanceLocalizationUpdate(
            enkf_truncation=1.0,
            rng=np.random.default_rng(7),
            param_type=Field,
            progress_callback=_noop,
        )
        std = StandardESUpdate(
            enkf_truncation=1.0,
            rng=np.random.default_rng(7),
            progress_callback=_noop,
        )
        if standard_first:
            std.prepare(ctx)
            dist.prepare(ctx)
        else:
            dist.prepare(ctx)
            std.prepare(ctx)
        assert dist._smoother is not None
        return np.asarray(dist._smoother.D_obs_minus_D).copy()

    innov_distance_first = _prepare_pair(standard_first=False)
    innov_standard_first = _prepare_pair(standard_first=True)

    np.testing.assert_allclose(
        innov_distance_first,
        innov_standard_first,
        err_msg=(
            "Distance smoother innovation depends on whether StandardESUpdate "
            "was prepared first — obs_context.responses is being mutated."
        ),
    )


@pytest.mark.parametrize(
    ("first_strategy_cls", "second_strategy_cls"),
    [
        (StandardESUpdate, DistanceLocalizationUpdate),
        (DistanceLocalizationUpdate, StandardESUpdate),
    ],
)
def test_that_obs_context_responses_survive_two_strategy_prepares(
    first_strategy_cls: type, second_strategy_cls: type
) -> None:
    ctx = _make_context()
    original = ctx.responses.copy()

    def _make(cls: type) -> object:
        if cls is StandardESUpdate:
            return StandardESUpdate(
                enkf_truncation=1.0,
                rng=np.random.default_rng(0),
                progress_callback=_noop,
            )
        return DistanceLocalizationUpdate(
            enkf_truncation=1.0,
            rng=np.random.default_rng(0),
            param_type=Field,
            progress_callback=_noop,
        )

    _make(first_strategy_cls).prepare(ctx)  # type: ignore[attr-defined]
    _make(second_strategy_cls).prepare(ctx)  # type: ignore[attr-defined]

    np.testing.assert_array_equal(ctx.responses, original)
