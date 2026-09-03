from abc import ABC, abstractmethod
from io import BytesIO

import xlsxwriter


class DesignInput(ABC):
    """Baseclass for creating test data mimicking Excel workbooks containing the 3
    sheets required by fmudesign in the form of a bytestream which can be passed
    to fmudesign.excel_to_dict().
    """

    GENERAL_SHEET: str = "general_input"
    DESIGN_SHEET: str = "designinput"
    DEFAULT_SHEET: str = "defaultvalues"

    @abstractmethod
    def excel_byte_stream(self) -> BytesIO:
        pass

    @abstractmethod
    def _write_general_input(self, wb: xlsxwriter.Workbook) -> None:
        pass

    @abstractmethod
    def _write_design_input(self, wb: xlsxwriter.Workbook) -> None:
        pass

    @abstractmethod
    def _write_default_values(self, wb: xlsxwriter.Workbook) -> None:
        pass
