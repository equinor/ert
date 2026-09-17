"""
Used for pre and processing of ert sensitivities, such as
setting up design matrix to run single sensitivities with ERT.
Output of this module can be used in custom standalone applications.
"""

import logging

from ._designsummary import summarize_design
from ._excel_to_dict import excel_to_dict, inputdict_to_yaml
from .create_design import DesignMatrix

logging.getLogger(__name__).addHandler(logging.NullHandler())
__all__ = [
    "DesignMatrix",
    "excel_to_dict",
    "inputdict_to_yaml",
    "summarize_design",
]
