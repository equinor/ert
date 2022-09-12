import numpy as np
import pytest

from ert._c_wrappers.enkf.export import (
    DesignMatrixReader,
    GenKwCollector,
    MisfitCollector,
    SummaryCollector,
)


def dumpDesignMatrix(path):
    with open(path, "w") as dm:
        dm.write(
            "REALIZATION	EXTRA_FLOAT_COLUMN EXTRA_INT_COLUMN EXTRA_STRING_COLUMN\n"
        )
        dm.write("0	0.08	125	ON\n")
        dm.write("1	0.07	225	OFF\n")
        dm.write("2	0.08	325	ON\n")
        dm.write("3	0.06	425	ON\n")
        dm.write("4	0.08	525	OFF\n")
        dm.write("5	0.08	625	ON\n")
        dm.write("6	0.09	725	ON\n")
        dm.write("7	0.08	825	OFF\n")
        dm.write("8	0.02	925	ON\n")
        dm.write("9	0.08	125	ON\n")
        dm.write("10	0.08	225	ON\n")
        dm.write("11	0.05	325	OFF\n")
        dm.write("12	0.08	425	ON\n")
        dm.write("13	0.07	525	ON\n")
        dm.write("14	0.08	625	UNKNOWN\n")
        dm.write("15	0.08	725	ON\n")
        dm.write("16	0.08	825	ON\n")
        dm.write("17	0.08	925	OFF\n")
        dm.write("18	0.09	125	ON\n")
        dm.write("19	0.08	225	ON\n")
        dm.write("20	0.06	325	OFF\n")
        dm.write("21	0.08	425	ON\n")
        dm.write("22	0.07	525	ON\n")
        dm.write("23	0.08	625	OFF\n")
        dm.write("24	0.08	725	ON\n")


def test_join(monkeypatch, snake_oil_case):
    ert = snake_oil_case
    monkeypatch.setenv("TZ", "CET")  # The ert_statoil case was generated in CET

    dumpDesignMatrix("DesignMatrix.txt")

    summary_data = SummaryCollector.loadAllSummaryData(ert, "default_1")
    gen_kw_data = GenKwCollector.loadAllGenKwData(ert, "default_1")
    misfit = MisfitCollector.loadAllMisfitData(ert, "default_1")
    dm = DesignMatrixReader.loadDesignMatrix("DesignMatrix.txt")

    result = summary_data.join(gen_kw_data, how="inner")
    result = result.join(misfit, how="inner")
    result = result.join(dm, how="inner")

    first_date = "2010-01-10"
    last_date = "2015-06-23"

    assert (
        pytest.approx(result["SNAKE_OIL_PARAM:OP1_OCTAVES"][0][first_date]) == 3.947766
    )
    assert (
        pytest.approx(result["SNAKE_OIL_PARAM:OP1_OCTAVES"][24][first_date]) == 4.206698
    )
    assert (
        pytest.approx(result["SNAKE_OIL_PARAM:OP1_OCTAVES"][24][last_date]) == 4.206698
    )
    assert pytest.approx(result["EXTRA_FLOAT_COLUMN"][0][first_date]) == 0.08

    assert result["EXTRA_INT_COLUMN"][0][first_date] == 125
    assert result["EXTRA_STRING_COLUMN"][0][first_date] == "ON"

    assert pytest.approx(result["EXTRA_FLOAT_COLUMN"][0][last_date]) == 0.08

    assert result["EXTRA_INT_COLUMN"][0][last_date] == 125
    assert result["EXTRA_STRING_COLUMN"][0][last_date] == "ON"

    assert pytest.approx(result["EXTRA_FLOAT_COLUMN"][1][last_date]) == 0.07

    assert result["EXTRA_INT_COLUMN"][1][last_date] == 225
    assert result["EXTRA_STRING_COLUMN"][1][last_date] == "OFF"

    assert pytest.approx(result["MISFIT:FOPR"][0][last_date]) == 457.500978
    assert pytest.approx(result["MISFIT:FOPR"][24][last_date]) == 1630.813862

    assert pytest.approx(result["MISFIT:TOTAL"][0][first_date]) == 468.479944
    assert pytest.approx(result["MISFIT:TOTAL"][0][last_date]) == 468.479944
    assert pytest.approx(result["MISFIT:TOTAL"][24][last_date]) == 1714.700855

    # pylint: disable=pointless-statement
    with pytest.raises(KeyError):
        # realization 13:
        result.loc[60]

    column_count = len(result.columns)
    assert result.dtypes[0] == np.float64
    assert result.dtypes[column_count - 1] == object
    assert result.dtypes[column_count - 2] == np.int64
