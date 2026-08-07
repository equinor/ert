from io import BytesIO

import xlsxwriter

from tests.ert.ui_tests.fmudesign.design_input.design_input import DesignInput


class FmudesignOneByOne(DesignInput):
    def excel_byte_stream(self) -> BytesIO:
        byte_stream = BytesIO()
        with xlsxwriter.Workbook(byte_stream) as wb:
            self._write_general_input(wb)
            self._write_design_input(wb)
            self._write_default_values(wb)
        byte_stream.seek(0)
        return byte_stream

    def _write_general_input(self, wb: xlsxwriter.Workbook) -> None:
        ws = wb.add_worksheet(self.GENERAL_SHEET)
        rows = [
            ["designtype", "onebyone"],
            ["repeats", 10],
            ["rms_seeds", "default"],
            ["background", None],
            ["distribution_seed", None],
        ]
        for row_idx, row in enumerate(rows):
            ws.write_row(row_idx, 0, row)

    def _write_design_input(self, wb: xlsxwriter.Workbook) -> None:
        ws = wb.add_worksheet(self.DESIGN_SHEET)
        header = [
            "sensname",
            "numreal",
            "type",
            "param_name",
            "senscase1",
            "value1",
            "senscase2",
            "value2",
            "dist_name",
            "dist_param1",
            "dist_param2",
            "dist_param3",
            "dist_param4",
            "decimals",
            "corr_sheet",
            "extern_file",
        ]
        rows = [
            [
                "rms_seed",
                None,
                "seed",
            ]
            + [None] * 13,
            [
                "faults",
                None,
                "scenario",
                "FAULT_POSITION",
                "east",
                -1,
                "west",
                1,
            ]
            + [None] * 8,
            [
                "velmodel",
                None,
                "scenario",
                "DC_MODEL",
                "alternative",
                "hum2",
            ]
            + [None] * 10,
            [
                "contacts",
                None,
                "scenario",
                "OWC1",
                "shallow",
                2600,
                "deep",
                2700,
            ]
            + [None] * 8,
            [
                None,
                None,
                None,
                "OWC2",
                None,
                2700,
                None,
                2800,
            ]
            + [None] * 8,
            [
                None,
                None,
                None,
                "OWC3",
                None,
                2800,
                None,
                2900,
            ]
            + [None] * 8,
            [
                "multz",
                20,
                "dist",
                "MULTZ_ILE",
                None,
                None,
                None,
                None,
                "logunif",
                0.0001,
                1,
            ]
            + [None] * 5,
        ]
        ws.write_row(0, 0, header)
        for row_idx, row in enumerate(rows, start=1):
            ws.write_row(row_idx, 0, row)

    def _write_default_values(self, wb: xlsxwriter.Workbook) -> None:
        ws = wb.add_worksheet(self.DEFAULT_SHEET)
        header = ["param_name", "default_values"]
        rows = [
            ["RMS_SEED", 1000],
            ["FAULT_POSITION", 0],
            ["DC_MODEL", "base"],
            ["OWC1", 2650],
            ["OWC2", 2750],
            ["OWC3", 2850],
            ["MULTZ_ILE", 0.1],
            ["PARAM1", 100],
            ["PARAM2", 200],
            ["PARAM3", 0.5],
        ]
        ws.write_row(0, 0, header)
        for row_idx, row in enumerate(rows, start=1):
            ws.write_row(row_idx, 0, row)
