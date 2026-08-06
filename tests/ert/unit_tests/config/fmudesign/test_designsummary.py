from pathlib import Path

from semeio.fmudesign import summarize_design

TESTDATA = Path(__file__).parent / "data"


def test_designsummary():
    """Test import and summary of design matrix"""

    snorrebergdesign = summarize_design(
        TESTDATA / "distributions/design.xlsx", "DesignSheet01"
    )
    # checking dimensions and some values in summary of design matrix
    assert snorrebergdesign.shape == (7, 9)

    assert (
        snorrebergdesign.columns
        == [
            "sensno",
            "sensname",
            "senstype",
            "casename1",
            "startreal1",
            "endreal1",
            "casename2",
            "startreal2",
            "endreal2",
        ]
    ).all()
    assert snorrebergdesign["sensname"][0] == "rms_seed"
    assert snorrebergdesign["senstype"][0] == "mc"
    assert snorrebergdesign["casename1"][0] == "P10_P90"
    assert snorrebergdesign["startreal1"][0] == 0
    assert snorrebergdesign["endreal1"][0] == 9
    assert snorrebergdesign["casename2"][0] is None
    assert snorrebergdesign["startreal2"][0] is None
    assert snorrebergdesign["endreal2"][0] is None

    assert snorrebergdesign["sensname"][6] == "relp_go"
    assert snorrebergdesign["senstype"][6] == "scalar"
    assert snorrebergdesign["casename1"][6] == "lc"
    assert snorrebergdesign["startreal1"][6] == 90
    assert snorrebergdesign["endreal1"][6] == 99
    assert snorrebergdesign["casename2"][6] == "hc"
    assert snorrebergdesign["startreal2"][6] == 100
    assert snorrebergdesign["endreal2"][6] == 109

    assert snorrebergdesign["endreal1"].sum() == 333

    # Test same also when design matrix is in .csv format
    designcsv = summarize_design(TESTDATA / "distributions/design.csv")
    assert snorrebergdesign.equals(designcsv)
