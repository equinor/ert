from pathlib import Path

import pandas as pd
import pytest

from fmudesign import DesignConfig
from fmudesign.design_input import DesignInput
from fmudesign.general_input import GeneralInput
from fmudesign.seed_strategy import SeedStrategy


def _general_input(
    rms_seeds=None,
    repeats=1,
    distribution_seed=None,
    seed_strategy=SeedStrategy.JOINT,
):
    return GeneralInput(
        designtype="onebyone",
        rms_seeds=rms_seeds,
        repeats=repeats,
        distribution_seed=distribution_seed,
        seed_strategy=seed_strategy,
    )


def _design_input(
    sensitivities=None,
    decimals=None,
):
    if sensitivities is None:
        sensitivities = {}
    if decimals is None:
        decimals = {}
    return DesignInput(sensitivities=sensitivities, decimals=decimals)


def _design_config(
    rms_seeds="default",
    repeats=10,
    distribution_seed=42,
    sensitivities=None,
    decimals=None,
    default_values=None,
):
    if default_values is None:
        default_values = pd.DataFrame()
    wb = "wb.xlsx"
    default_sheet = "default"
    with pd.ExcelWriter(wb, engine="openpyxl") as writer:
        default_values.to_excel(
            writer, sheet_name=default_sheet, index=False, header=None
        )
    gi = _general_input(
        rms_seeds=rms_seeds, repeats=repeats, distribution_seed=distribution_seed
    )
    di = _design_input(sensitivities=sensitivities, decimals=decimals)
    return DesignConfig.from_input(
        input_filename=wb,
        general_input=gi,
        design_input=di,
        default_values_sheet=default_sheet,
    )


def test_that_rms_seeds_file_with_only_newline_raises_value_error(use_tmpdir):
    seeds_file = "seeds.txt"
    Path(seeds_file).write_text("\n", encoding="utf-8")
    with pytest.raises(
        ValueError, match=rf"rms_seeds file '{seeds_file}' contains no data."
    ):
        _design_config(rms_seeds=seeds_file)
