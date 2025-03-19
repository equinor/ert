import logging
from pathlib import Path

from .design_matrix import DesignMatrix

logger = logging.getLogger(__name__)


class FMLogger:
    @staticmethod
    def validate_ert_design_matrix(
        xlsfilename: Path, designsheetname: str, defaultssheetname: str
    ) -> DesignMatrix | None:
        try:
            return DesignMatrix(xlsfilename, designsheetname, defaultssheetname)
        except Exception as exc:
            logger.warning(
                f"DESIGN_MATRIX validation of DESIGN2PARAMS would have failed with: {exc!s}"
            )
