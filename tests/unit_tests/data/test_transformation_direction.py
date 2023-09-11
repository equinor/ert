import pytest

from ert.data import TransformationDirection


def test_transformation_direction_str():
    assert str(TransformationDirection.FROM_RECORD) == "from_record"
    assert str(TransformationDirection.TO_RECORD) == "to_record"
    assert str(TransformationDirection.BIDIRECTIONAL) == "bidirectional"
    assert str(TransformationDirection.NONE) == "none"
    with pytest.raises(ValueError):
        _ = str(TransformationDirection.NONE | TransformationDirection.FROM_RECORD)
