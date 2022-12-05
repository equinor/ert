"""
Tests that config parsing works as intended from the view point of an ert developer.
"""
import pytest
from hypothesis import given

from ert._c_wrappers.enkf import ResConfig
from tests.test_config_parsing.config_dict_generator import config_dicts, to_config_file


def touch(filename):
    with open(filename, "w", encoding="utf-8") as fh:
        fh.write(" ")


@pytest.fixture()
def set_site_config(monkeypatch, tmp_path):
    # GIVEN that the site config does not set any values
    test_site_config = tmp_path / "test_site_config.ert"
    test_site_config.write_text("\n")
    monkeypatch.setenv("ERT_SITE_CONFIG", str(test_site_config))


@pytest.mark.usefixtures("use_tmpdir", "set_site_config")
@given(config_dicts())
@pytest.mark.xfail(reason="https://github.com/equinor/ert/issues/4178")
def test_res_config_simple_config_parsing(config_dict):
    # AS AN ert developer

    # GIVEN any config file that is to be read
    filename = "config.ert"

    # AND GIVEN a config_dict with the same contents
    to_config_file(filename, config_dict)

    # THEN reading that file results in the same configuration
    filename = "config.ert"
    to_config_file(filename, config_dict)
    assert ResConfig("test.ert") == ResConfig(config_dict=config_dict)
