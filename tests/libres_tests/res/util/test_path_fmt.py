from ert._c_wrappers.util import PathFormat


def test_create():
    path_fmt = PathFormat("random/path/%d-%d")
    assert "random/path" in repr(path_fmt)
    assert str(path_fmt).startswith("PathFormat(")
