import logging
from collections.abc import Sequence
from pathlib import Path

from .design_matrix import DesignMatrix

logger = logging.getLogger(__name__)


class FMLogger:
    def validate_ert_design_matrix(
        self, xlsfilename: Path, designsheetname: str, defaultssheetname: str
    ) -> DesignMatrix | None:
        try:
            return DesignMatrix(xlsfilename, designsheetname, defaultssheetname)
        except Exception as exc:
            logger.warning(
                f"DESIGN_MATRIX validation of DESIGN2PARAMS would have failed with: {exc!s}"
            )

    def validate_design_matrix_merge(
        self, design_matrices: Sequence[DesignMatrix]
    ) -> None:
        try:
            main_design_matrix: DesignMatrix | None = None
            for design_matrix in design_matrices:
                if main_design_matrix is None:
                    main_design_matrix = design_matrix
                else:
                    main_design_matrix = main_design_matrix.merge_with_other(
                        design_matrix
                    )
        except Exception as exc:
            logger.warning(f"Design matrix merging would have failed due to: {exc}")
