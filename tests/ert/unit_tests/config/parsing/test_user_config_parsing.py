import pytest

from ert.config import ErtConfig
from ert.config.parsing import (
    ConfigValidationError,
    init_user_config_schema,
    parse_contents,
)


def test_that_rft_entry_is_parsable():
    parsed = parse_contents(
        """
        NUM_REALIZATIONS 1

        RFT WELL:NAME DATE:2020-12-13
        """,
        init_user_config_schema(),
        "unused",
    )

    del parsed["DEFINE"]

    assert parsed == {
        "NUM_REALIZATIONS": 1,
        "RFT": [{"WELL": "NAME", "DATE": "2020-12-13"}],
    }


def test_that_rft_entry_is_a_multi_occurrence_keyword():
    parsed = parse_contents(
        """
        NUM_REALIZATIONS 1

        RFT WELL:NAME1 DATE:2020-12-13
        RFT WELL:NAME2 DATE:2021-11-14
        """,
        init_user_config_schema(),
        "unused",
    )

    del parsed["DEFINE"]

    assert parsed == {
        "NUM_REALIZATIONS": 1,
        "RFT": [
            {"WELL": "NAME1", "DATE": "2020-12-13"},
            {"WELL": "NAME2", "DATE": "2021-11-14"},
        ],
    }


def test_that_es_mda_weights_is_parsable():
    parsed = parse_contents(
        """
        NUM_REALIZATIONS 1

        ANALYSIS_SET_VAR STD_ENKF WEIGHTS 8, 4, 2, 1
        """,
        init_user_config_schema(),
        "unused",
    )

    del parsed["DEFINE"]

    assert parsed == {
        "NUM_REALIZATIONS": 1,
        "ANALYSIS_SET_VAR": [["STD_ENKF", "WEIGHTS", "8, 4, 2, 1"]],
    }


def test_that_es_mda_weights_is_stored_in_analysis_config():
    config = ErtConfig.from_file_contents(
        """
        NUM_REALIZATIONS 1
        ANALYSIS_SET_VAR STD_ENKF WEIGHTS 8, 4, 2, 1
        """
    )

    assert config.analysis_config.es_settings.weights == "8, 4, 2, 1"
    assert config.analysis_config.es_settings.weights_from_config


def test_that_invalid_es_mda_weights_fails_validation():
    with pytest.raises(ConfigValidationError, match="Invalid weights: 0"):
        ErtConfig.from_file_contents(
            """
            NUM_REALIZATIONS 1
            ANALYSIS_SET_VAR STD_ENKF WEIGHTS 0
            """
        )
