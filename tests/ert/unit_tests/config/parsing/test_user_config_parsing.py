from ert.config.parsing import init_user_config_schema, parse_contents


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
        "RFT": [[{"WELL": "NAME", "DATE": "2020-12-13"}]],
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
            [{"WELL": "NAME1", "DATE": "2020-12-13"}],
            [{"WELL": "NAME2", "DATE": "2021-11-14"}],
        ],
    }
