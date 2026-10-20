"""Module for summarizing design set up for one by one sensitivities"""

import pandas as pd

from .utils import read_excel_sheet


def _get_sensitivity_type(senscase: str) -> str:
    """Determine sensitivity type based on the case name"""
    sensitivity_types = {"p10_p90": "mc", "ref": "ref"}
    return sensitivity_types.get(senscase.lower(), "scalar")


def summarize_design(filename: str, sheetname: str = "DesignSheet01") -> pd.DataFrame:
    """
     Summarizes the design set up for one by one sensitivities
     specified in a design matrix on standard fmu format.

    Args:
        filename (str): Name of excel or csv file containing designmatrix
            for one by one sensitivities on standard FMU format.
        sheetname (str): Name of sheet in excel workbook which
            contains the designmatrix (only for excel input). Defaults to
            'DesignSheet01'.

    Returns:
        pd.DataFrame: Summary of sensitivities,
        corresponding realisation numbers,
        senstype('mc' or 'scalar')
        and senscase (name of high and low cases).
        Each row represents one sensitivity
        with 1-2 cases (low/high).
        Column names are ['sensno', 'sensname',
        'senstype', 'casename1', 'startreal1', 'endreal1',
        'casename2', 'startreal2', 'endreal2']

    Example:
        >> from fmudesign import summarize_design
        >> designname = 'design_filename.xlsx'
        >> designsheet = 'DesignSheet01'
        >> designtable = summarize_design(designname, designsheet)

    """

    # Read design matrix
    if str(filename).endswith(".xlsx"):
        dgn = read_excel_sheet(filename, sheetname)
    elif str(filename).endswith(".csv"):
        dgn = pd.read_csv(filename)
    else:
        raise ValueError(
            "Design matrix must be on Excel or csv format"
            " and filename must end with .xlsx or .csv"
        )

    # Initialize results DataFrame with same columns
    designsummary = pd.DataFrame(
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
        ]
    )

    for sensno, (sensname, sens_group) in enumerate(
        dgn.groupby("SENSNAME", sort=False)
    ):
        # Get cases in order of appearance
        cases = sens_group["SENSCASE"].unique()
        summary = [sensno, sensname, _get_sensitivity_type(cases[0])]
        for case in cases[:2]:
            reals = sens_group.loc[sens_group["SENSCASE"] == case, "REAL"]
            summary += [case, reals.min(), reals.max()]
        if len(cases) == 1:
            summary += [None, None, None]
        designsummary.loc[sensno] = summary

    return designsummary
