import pytest

from ert.config.parsing import init_user_config_schema, parse_contents
from ert.config.parsing.config_errors import ConfigValidationError


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


def test_that_seismic_entries_are_parsable():
    parsed = parse_contents(
        """
        NUM_REALIZATIONS 1

        SEISMIC tables/field--amplitude_full_mean_depth--20250101_20240101.csv
        SEISMIC tables/field--amplitude_full_min_depth--20250101_20240101.csv
        """,
        init_user_config_schema(),
        "unused",
    )

    del parsed["DEFINE"]

    assert parsed == {
        "NUM_REALIZATIONS": 1,
        "SEISMIC": [
            "tables/field--amplitude_full_mean_depth--20250101_20240101.csv",
            "tables/field--amplitude_full_min_depth--20250101_20240101.csv",
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
