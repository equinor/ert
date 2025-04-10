import logging
from pathlib import Path

from .design_matrix import DesignMatrix

logger = logging.getLogger(__name__)


class DesignMatrixValidator:
    def __init__(self) -> None:
        self.design_matrices: list[DesignMatrix] = []

    def validate_design_matrix(self, private_args: dict[str, str]) -> None:
        try:
            xlsfilename = Path(private_args["<xls_filename>"])
            designsheet = private_args["<designsheet>"]
            defaultsheet = private_args["<defaultssheet>"]
            self.design_matrices.append(
                DesignMatrix(xlsfilename, designsheet, defaultsheet)
            )
        except Exception as exc:
            logger.warning(
                "DESIGN_MATRIX validation of DESIGN2PARAMS would have "
                f"failed with: {exc!s}"
            )

    def validate_design_matrix_merge(self) -> None:
        try:
            main_design_matrix: DesignMatrix | None = None
            for design_matrix in self.design_matrices:
                if main_design_matrix is None:
                    main_design_matrix = design_matrix
                else:
                    main_design_matrix.merge_with_other(design_matrix)
        except Exception as exc:
            logger.warning(f"Design matrix merging would have failed due to: {exc}")
