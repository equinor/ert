import pytest

from ert.config import SamplerConfig


@pytest.mark.parametrize(
    "backend, method, expected",
    [
        (None, "norm", "norm"),
        ("scipy", None, "scipy/norm"),
        ("scipy", "norm", "scipy/norm"),
    ],
)
def test_sampler_config_backend_and_method(backend, method, expected):
    config_dict = {}
    if backend is not None:
        config_dict["backend"] = backend
    if method is not None:
        config_dict["method"] = method
    config = SamplerConfig.model_validate(config_dict)
    assert config.backend is None
    assert config.method == expected


@pytest.mark.parametrize(
    "backend, method, expected",
    [
        (None, "foo", "Sampler method 'foo' not found"),
        (None, "default", "Cannot specify 'default' method without a plugin name"),
        ("foo", None, "Sampler method 'foo/default' not found"),
        ("foo", "bar", "Sampler method 'foo/bar' not found"),
    ],
)
def test_sampler_config_backend_and_method_errors(backend, method, expected):
    config_dict = {}
    if backend is not None:
        config_dict["backend"] = backend
    if method is not None:
        config_dict["method"] = method
    with pytest.raises(ValueError, match=expected):
        SamplerConfig.model_validate(config_dict)
