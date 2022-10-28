import datetime
import os
import os.path

import pytest

from ert._c_wrappers.enkf import ConfigKeys, ResConfig, SubstConfig


def with_key(a_dict, key, val):
    copy = a_dict.copy()
    copy[key] = val
    return copy


def without_key(a_dict, key):
    return {index: value for index, value in a_dict.items() if index != key}


@pytest.fixture(name="snake_oil_structure_config")
def fixture_snake_oil_structure_config(copy_case):
    copy_case("snake_oil_structure")
    cwd = os.getcwd()
    config_file_name = "config"
    return {
        ConfigKeys.RUNPATH_FILE: "runpath",
        pytest.TEST_CONFIG_FILE_KEY: config_file_name,
        ConfigKeys.DEFINE_KEY: {
            "<CWD>": cwd,
            "<CONFIG_PATH>": cwd,
            "<CONFIG_FILE>": config_file_name,
            "<CONFIG_FILE_BASE>": config_file_name,
            "keyA": "valA",
            "keyB": "valB",
        },
        ConfigKeys.DATA_KW_KEY: {"keyC": "valC", "keyD": "valD"},
    }


@pytest.fixture(name="snake_oil_structure_config_file")
def fixture_snake_oil_structure_config_file(snake_oil_structure_config):
    filename = snake_oil_structure_config[pytest.TEST_CONFIG_FILE_KEY]
    with open(file=filename, mode="w+", encoding="utf-8") as config:
        # necessary in the file, but irrelevant to this test
        config.write("JOBNAME  Job%d\n")
        config.write("NUM_REALIZATIONS  1\n")

        # write the rest of the relevant config items to the file
        config.write(
            f"{ConfigKeys.RUNPATH_FILE} "
            f"{snake_oil_structure_config[ConfigKeys.RUNPATH_FILE]}\n"
        )
        defines = snake_oil_structure_config[ConfigKeys.DEFINE_KEY]
        for key in defines:
            val = defines[key]
            config.write(f"{ConfigKeys.DEFINE_KEY} {key} {val}\n")
        data_kws = snake_oil_structure_config[ConfigKeys.DATA_KW_KEY]
        for key in data_kws:
            val = data_kws[key]
            config.write(
                f"{ConfigKeys.DATA_KW_KEY} {key} {val}\n".format(
                    ConfigKeys.DATA_KW_KEY, key, val
                )
            )

    return filename


def test_two_instances_of_same_config_are_equal(snake_oil_structure_config):
    assert SubstConfig.from_dict(
        config_dict=snake_oil_structure_config, num_cpu=1
    ) == SubstConfig.from_dict(config_dict=snake_oil_structure_config, num_cpu=1)


def test_two_instances_of_different_config_are_not_equal(snake_oil_structure_config):
    assert SubstConfig.from_dict(
        config_dict=snake_oil_structure_config, num_cpu=1
    ) != SubstConfig.from_dict(
        config_dict=with_key(
            snake_oil_structure_config, ConfigKeys.RUNPATH_FILE, "aaaaa"
        ),
        num_cpu=1,
    )


def test_old_and_new_constructor_creates_equal_config(
    snake_oil_structure_config_file, snake_oil_structure_config
):
    assert ResConfig(
        user_config_file=snake_oil_structure_config_file
    ).subst_config == SubstConfig.from_dict(
        config_dict=snake_oil_structure_config, num_cpu=1
    )


def test_complete_config_reads_correct_values(snake_oil_structure_config):
    subst_config = SubstConfig.from_dict(
        config_dict=snake_oil_structure_config, num_cpu=1
    )
    assert subst_config["<CWD>"] == os.getcwd()
    assert subst_config["<CONFIG_PATH>"] == os.getcwd()
    assert subst_config["<DATE>"] == datetime.datetime.now().date().isoformat()
    assert subst_config["keyA"] == "valA"
    assert subst_config["keyB"] == "valB"
    assert subst_config["keyC"] == "valC"
    assert subst_config["keyD"] == "valD"
    assert subst_config["<RUNPATH_FILE>"] == os.getcwd() + "/runpath"
    assert subst_config["<NUM_CPU>"] == "1"


def test_missing_runpath_gives_default_value(snake_oil_structure_config):
    subst_config = SubstConfig.from_dict(
        config_dict=without_key(snake_oil_structure_config, ConfigKeys.RUNPATH_FILE),
        num_cpu=1,
    )
    assert subst_config["<RUNPATH_FILE>"] == os.getcwd() + "/.ert_runpath_list"
