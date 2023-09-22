import numpy as np
import pytest

from ert.run_models import MultipleDataAssimilation as mda


def test_normalized_weights():
    weights = mda.normalizeWeights([1])
    assert weights == [1.0]

    weights = mda.normalizeWeights([1, 1])
    assert weights == [2.0, 2.0]

    weights = np.array(mda.normalizeWeights([8, 4, 2, 1]))
    assert np.reciprocal(weights).sum() == 1.0


def test_weights():
    weights = mda.parseWeights("2, 2, 2, 2")
    assert weights == [2, 2, 2, 2]

    weights = mda.parseWeights("1, 2, 3, ")
    assert weights == [1, 2, 3]

    weights = mda.parseWeights("1, 0, 1")
    assert weights == [1, 1]

    weights = mda.parseWeights("1.414213562373095, 1.414213562373095")
    assert weights == [1.414213562373095, 1.414213562373095]

    with pytest.raises(ValueError):
        mda.parseWeights("2, error, 2, 2")
