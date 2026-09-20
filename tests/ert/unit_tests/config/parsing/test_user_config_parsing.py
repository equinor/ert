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


def test_that_es_mda_weights_are_parsable():
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


def test_that_es_mda_weights_are_stored_in_analysis_config():
    config = ErtConfig.from_file_contents(
        """
        NUM_REALIZATIONS 1
        ANALYSIS_SET_VAR STD_ENKF WEIGHTS 8, 4, 2, 1
        """
    )

    assert config.analysis_config.es_settings.weights == "8, 4, 2, 1"


def test_that_invalid_es_mda_weights_fail_validation():
    with pytest.raises(ConfigValidationError, match="Invalid weights: 0"):
        ErtConfig.from_file_contents(
            """
            NUM_REALIZATIONS 1
            ANALYSIS_SET_VAR STD_ENKF WEIGHTS 0
            """
        )


def test_that_seismic_entries_are_parsable():
    parsed = parse_contents(
        """
        NUM_REALIZATIONS 1

        SEISMIC tables/horizon--amplitude_full_mean_depth--20250101_20240101.csv
        SEISMIC tables/horizon--amplitude_full_min_depth--20250101_20240101.csv
        """,
        init_user_config_schema(),
        "unused",
    )

    del parsed["DEFINE"]

    assert parsed == {
        "NUM_REALIZATIONS": 1,
        "SEISMIC": [
            "tables/horizon--amplitude_full_mean_depth--20250101_20240101.csv",
            "tables/horizon--amplitude_full_min_depth--20250101_20240101.csv",
        ],
    }


def test_that_seismic_entry_is_limited_to_one_element():
    with pytest.raises(
        ConfigValidationError, match="SEISMIC must have maximum 1 arguments"
    ):
        parse_contents(
            """
            NUM_REALIZATIONS 1

            SEISMIC first.csv second.csv
            """,
            init_user_config_schema(),
            "unused",
        )
