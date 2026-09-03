import math

import pytest

from ert.config.analysis_module import ESSettings


@pytest.mark.parametrize(
    ("settings", "ensemble_size", "expected_threshold"),
    [
        (ESSettings(), 8, 3 / math.sqrt(8)),
        (ESSettings(), 9, 1.0),
        (ESSettings(), 36, 0.5),
        (ESSettings(), 200, 3 / math.sqrt(200)),
        (ESSettings(localization_correlation_threshold=0.2), 9, 0.2),
    ],
)
def test_that_correlation_threshold_uses_default_or_custom_value(
    settings, ensemble_size, expected_threshold
):
    assert settings.correlation_threshold(ensemble_size) == pytest.approx(
        expected_threshold
    )
