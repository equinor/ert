from res.enkf import SummaryKeyMatcher

from ...libres_utils import ResTest


def test_creation():
    matcher = SummaryKeyMatcher()

    assert len(matcher) == 0

    matcher.addSummaryKey("F*")
    assert len(matcher) == 1

    matcher.addSummaryKey("F*")
    assert len(matcher) == 1

    matcher.addSummaryKey("FOPT")
    assert len(matcher) == 2

    assert list(matcher.keys()) == ["FOPT", "F*"]

    assert "FGIR" in matcher
    assert "FOPT" in matcher
    assert "TCPU" not in matcher

    assert matcher.isRequired("FOPT")
    assert not matcher.isRequired("FGIR")
    assert not matcher.isRequired("TCPU")
