import pytest


def is_equal_list(left: dict, right: dict, path, errors):
    equal = True
    if not len(left) == len(right):
        errors.append(
            f"At {path}, the lists differ in length. "
            f"Expected size {len(left)}, actual {len(right)}")
        equal = False

    for i, l_val in enumerate(left):
        if len(right) > i:
            r_val = right[i]
            if not check_val(l_val, r_val, f"{path}[{i}]", errors):
                equal = False
    return equal


def is_equal_dict(left: dict, right: dict, path, errors):
    equal = True
    if not set(left.keys()) == set(right.keys()):
        expected = set(left.keys()).difference(set(right.keys()))
        unexpected = set(right.keys()).difference(set(left.keys()))
        error = f"At {path}"
        if expected:
            error += f", missing {expected}"
        if unexpected:
            error += f", unexpected {unexpected}"
        errors.append(error)
        equal = False

    for k, l_val in left.items():
        if k in right:
            r_val = right.get(k)
            if not check_val(l_val, r_val, f"{path}/{k}", errors):
                equal = False
    return equal


def check_val(l_val, r_val, path, errors):
    if isinstance(l_val, dict):
        if isinstance(r_val, dict):
            return is_equal_dict(l_val, r_val, path, errors)
        else:
            errors.append(f"At {path}, expected type (dict), actual type ({type(r_val)})")
            return False
    elif isinstance(l_val, list):
        if isinstance(r_val, list):
            return is_equal_list(l_val, r_val, path, errors)
        else:
            errors.append(f"At {path}, expected type (list), actual type ({type(r_val)})")
            return False
    elif not l_val == r_val:
        if isinstance(r_val, (dict, list)):
            error = f"expected type ({type(l_val)}), actual type ({type(r_val)})"
        else:
            error = f"expected value ({l_val}), actual value ({r_val})"
        errors.append(f"At {path}, {error}")
        return False
    return True


def assert_equal_dicts(left, right):
    errors = []
    path = ""
    if not is_equal_dict(left, right, path, errors):
        raise AssertionError("Dicts are not equal: \n" + "\n".join(errors))
    assert left == right, "There is a bug in the custom comparison code above"


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ERT_STORAGE_DATABASE_URL", "sqlite://")
    monkeypatch.setenv("ERT_STORAGE_NO_TOKEN", "yup")


@pytest.fixture
def ert_storage_app(env):
    from ert_storage.app import app

    return app


@pytest.fixture
def dark_storage_app(env):
    from ert_shared.dark_storage.app import app

    return app


def test_openapi(ert_storage_app, dark_storage_app):
    """
    Test that the openapi.json of Dark Storage is identical to ERT Storage
    """
    expect = ert_storage_app.openapi()
    actual = dark_storage_app.openapi()

    # Remove textual data (descriptions and such) from ERT Storage's API.
    def _remove_text(data):
        if isinstance(data, dict):
            return {
                key: _remove_text(val)
                for key, val in data.items()
                if key not in ("description", "examples")
            }
        return data

    assert_equal_dicts(_remove_text(expect), _remove_text(actual))
