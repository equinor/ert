import pathlib
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


@pytest.mark.usefixtures("use_tmpdir")
def test_that_convert_summary_observations_produces_natsorted_csv_rows(capsys):
    def make_summary_obs(number: int):
        return {
            f"WOPR:OP{number}": [
                {
                    "name": f"WOPR_OP{number}",
                    "errors": [0.02],
                    "values": [0.5],
                    "x_axis": ["2010-01-27T00:00:00"],
                },
            ]
        }

    summary_obs = (
        make_summary_obs(30)
        | make_summary_obs(10)
        | make_summary_obs(4)
        | make_summary_obs(2)
    )
    convert_summary_observations(
        summary_observations=summary_obs,
        breakthrough_observations={},
        csv_file_name="foo.csv",
    )
    with pathlib.Path("foo.csv").open(encoding="utf-8") as f:
        csv_content = f.readlines()
    well_names = [row.split(",")[1].strip() for row in csv_content[1:]]
    assert well_names == ["OP2", "OP4", "OP10", "OP30"]
