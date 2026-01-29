from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
from resfo_utilities import RFTReader

from ert.mode_definitions import ENSEMBLE_SMOOTHER_MODE

from .run_cli import run_cli


@pytest.mark.slow
@pytest.mark.skipif(not shutil.which("flow"), reason="flow not available")
def test_that_rft_example_with_rft_observation_keyword_yelds_same_result_as_gendata_rft(
    use_tmpdir, source_root, snapshot, request
):
    """
    Created snapshot of rft pressures by using the GENDATA_RFT in combination
    with GENERAL_OBSERVATION. This test runs the same experiment using RFT_OBSERVATION
    to verify that the resulting RFT pressure data is the same.
    """
    shutil.copytree(
        os.path.join(source_root, "test-data", "ert", "rft_example"), "test-data"
    )
    run_cli(ENSEMBLE_SMOOTHER_MODE, "--disable-monitoring", "test-data/rft.ert")
    pressure = {}
    for file in sorted(Path("spe1_out").rglob("*.RFT")):
        rft = RFTReader.open(file)
        for entry in rft:
            key = "/".join(file.parts[-3:-1]) + f", {entry.date}, {entry.well}"
            pressure[key] = entry["PRESSURE"].tolist()

    FILE_NAME = "rft_pressures.json"

    if bool(request.config.getoption("--snapshot-update")):
        snapshot.assert_match(json.dumps(pressure, indent=2) + "\n", FILE_NAME)

    with open(Path(snapshot.snapshot_dir, FILE_NAME), encoding="utf-8") as file:
        pressure_snapshot = json.load(file)

    assert isinstance(pressure, dict)
    for key, snapshot_value in pressure_snapshot.items():
        assert pressure.get(key) == pytest.approx(snapshot_value, abs=0.05), (
            f"RFT pressure snapshot test failed for {key}"
        )
