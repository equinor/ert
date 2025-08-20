import hypothesis.strategies as st
import pytest
from hypothesis import given
from resdata.summary import Summary

from ert.summary_key_type import is_rate
from tests.ert.unit_tests.config.summary_generator import summary_variables


def nonempty_string_without_whitespace():
    return st.text(
        st.characters(whitelist_categories=("Lu", "Ll", "Nd", "P")), min_size=1
    )


@given(key=nonempty_string_without_whitespace())
def test_is_rate_does_not_raise_error(key):
    is_rate_bool = is_rate(key)
    assert isinstance(is_rate_bool, bool)


key_rate_examples = [
    ("OPR", False),
    ("WOPR:OP_4", True),
    ("WGIR", True),
    ("FOPT", False),
    ("GGPT", False),
    ("RWPT", False),
    ("COPR", True),
    ("LPR", False),
    ("LWPR", False),
    ("LCOPR", True),
    ("RWGIR", True),
    ("RTPR", True),
    ("RXFR", True),
    ("XXX", False),
    ("YYYY", False),
    ("ZZT", False),
    ("SGPR", False),
    ("AAPR", False),
    ("JOPR", False),
    ("ROPRT", True),
    ("RNFT", False),
    ("RFR", False),
    ("RRFRT", True),
    ("ROC", False),
    ("BPR:123", False),
    ("FWIR", True),
]


@pytest.mark.parametrize("key, rate", key_rate_examples)
def test_is_rate_determines_rate_key_correctly(key, rate):
    is_rate_bool = is_rate(key)
    assert is_rate_bool == rate


@given(key=summary_variables())
def test_rate_determination_is_consistent(key):
    # Here we verify that the determination of rate keys is the same
    # as provided by resdata api
    assert Summary.is_rate(key) == is_rate(key)
