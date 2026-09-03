from pathlib import Path

import pandas as pd
import pytest

from fmudesign import summarize_design

DESIGN = pd.DataFrame(
    {
        "REAL": range(7),
        "SENSNAME": ["rms_seed"] * 2 + ["faults"] * 4 + ["reference"],
        "SENSCASE": ["P10_P90"] * 2 + ["low"] * 2 + ["high"] * 2 + ["ref"],
    }
)

EXPECTED_SUMMARY = pd.DataFrame(
    [
        [0, "rms_seed", "mc", "P10_P90", 0, 1, None, None, None],
        [1, "faults", "scalar", "low", 2, 3, "high", 4, 5],
        [2, "reference", "ref", "ref", 6, 6, None, None, None],
    ],
    columns=[
        "sensno",
        "sensname",
        "senstype",
        "casename1",
        "startreal1",
        "endreal1",
        "casename2",
        "startreal2",
        "endreal2",
    ],
    dtype=object,
)


@pytest.mark.parametrize("file_format", ["csv", "xlsx"])
def test_summarize_design(tmp_path: Path, file_format: str):
    design_path = tmp_path / f"design.{file_format}"
    if file_format == "csv":
        DESIGN.to_csv(design_path, index=False)
        summary = summarize_design(design_path)
    else:
        DESIGN.to_excel(design_path, sheet_name="design", index=False)
        summary = summarize_design(design_path, "design")

    pd.testing.assert_frame_equal(summary, EXPECTED_SUMMARY, check_dtype=False)
