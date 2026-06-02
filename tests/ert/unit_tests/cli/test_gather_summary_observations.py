from textwrap import dedent

import pytest

from ert.gather_summary_observations import convert_summary_observations


@pytest.mark.usefixtures("use_tmpdir")
def test_that_convert_summary_observations_extracts_localization_information(capsys):
    summary_obs = {
        "WOPR:WELL_WITH_LOCALIZATION": [
            {
                "name": "WOPR_OP1_72",
                "errors": [0.02],
                "values": [0.5],
                "east": [None],
                "north": [None],
                "radius": [None],
                "x_axis": ["2010-01-27T00:00:00"],
            },
            {
                "name": "WOPR_OP1_9",
                "errors": [0.05],
                "values": [0.1],
                "east": [10],
                "north": [20],
                "radius": [2500],
                "x_axis": ["2010-03-31T00:00:00"],
            },
        ],
        "WSPC:WELL_WITHOUT_RADIUS": [
            {
                "name": "WSPC",
                "errors": [0.04],
                "values": [3],
                "east": [40],
                "north": [50],
                "radius": [None],
                "x_axis": ["2011-12-21T00:00:00"],
            },
        ],
        "WSIR:WELL_WITHOUT_LOCALIZATION": [
            {
                "name": "WSIR",
                "errors": [0.04],
                "values": [3],
                "x_axis": ["2011-12-21T00:00:00"],
            },
        ],
    }
    convert_summary_observations(
        summary_observations=summary_obs,
        breakthrough_observations={},
        csv_file_name="foo.csv",
    )
    expected_print_with_localization = dedent("""\
    SUMMARY {
      VALUES = foo.csv;
      WELL WELL_WITHOUT_RADIUS {
        LOCALIZATION {
          EAST=40;
          NORTH=50;
        };
      };
      WELL WELL_WITH_LOCALIZATION {
        LOCALIZATION {
          EAST=10;
          NORTH=20;
          RADIUS=2500;
        };
      };
    };""")

    assert expected_print_with_localization in capsys.readouterr().out
