import pytest

from ert.storage.migration._block_fs_native import Kind, parse_name


@pytest.mark.parametrize(
    "name,expect",
    [
        ("foo.0", ("foo", 0, 0)),
        ("bar.1.2", ("bar.1", 0, 2)),
        ("foo bar baz.123.456", ("foo bar baz.123", 0, 456)),
        ("a.b.c.d.e.0.1", ("a.b.c.d.e.0", 0, 1)),
    ],
)
def test_parse_summary_name(name, expect):
    actual = parse_name(name, Kind.SUMMARY)
    assert actual == expect


@pytest.mark.parametrize("name", ["", "foo", "foo.one", "foo.one.two", "foo.0.two"])
def test_parse_summary_name_fail(name):
    with pytest.raises(ValueError):
        parse_name(name, Kind.SUMMARY)


@pytest.mark.parametrize(
    "name,expect",
    [
        ("foo.0.1", ("foo", 0, 1)),
        ("bar.1.2", ("bar", 1, 2)),
        ("foo bar baz.123.456", ("foo bar baz", 123, 456)),
        ("a.b.c.d.e.0.1", ("a.b.c.d.e", 0, 1)),
    ],
)
def test_parse_name(name, expect):
    # All kind except for SUMMARY parse the same way. Use GEN_KW as placeholder
    actual = parse_name(name, Kind.GEN_KW)
    assert actual == expect


@pytest.mark.parametrize(
    "name", ["", "foo", "foo.one", "foo.one.two", "foo.one.0", "foo.0.two"]
)
def test_parse_name_fail(name):
    # All kind except for SUMMARY parse the same way. Use GEN_KW as placeholder
    with pytest.raises(ValueError):
        parse_name(name, Kind.GEN_KW)
