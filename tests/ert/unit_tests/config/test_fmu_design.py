from pathlib import Path
from textwrap import dedent

import openpyxl

from ert.config import ErtConfig


def write_fmu_design_workbook(
    path: str | Path,
    *,
    design_sheet: str = "designinput",
    default_sheet: str = "defaultvalues",
) -> None:
    sheets = {
        "general_input": dedent(
            """\
            designtype,onebyone
            repeats,1
            rms_seeds,None
            background,None
            distribution_seed,None"""
        ),
        design_sheet: dedent(
            """\
            sensname,numreal,type,param_name,senscase1,value1,senscase2,value2
            faults,,scenario,FAULT_POSITION,east,-1,west,1"""
        ),
        default_sheet: dedent(
            """\
            param_name,default_values
            FAULT_POSITION,0
            OWC,2650"""
        ),
    }

    workbook = openpyxl.Workbook(write_only=True)
    for sheet_name, csv in sheets.items():
        sheet = workbook.create_sheet(sheet_name)
        for line in csv.splitlines():
            sheet.append(line.split(","))
    workbook.save(path)


def parse_ert_config_with_fmu_design(fmu_design_line: str) -> ErtConfig:
    Path("config.ert").write_text(
        f"""
        NUM_REALIZATIONS 2
        {fmu_design_line}
        """,
        encoding="utf-8",
    )
    return ErtConfig.from_file("config.ert")
