from __future__ import annotations

from typing import TYPE_CHECKING

import ert

from .csv_export import CSVExportJob
from .disable_parameters import DisableParametersUpdate
from .export_misfit_data import ExportMisfitDataJob
from .export_runpath import ExportRunpathJob
from .gen_data_rft_export import GenDataRFTCSVExportJob
from .misfit_preprocessor import MisfitPreprocessor

if TYPE_CHECKING:
    from ert.plugins.workflow_config import WorkflowConfigs


@ert.plugin(name="ert")
def legacy_ertscript_workflow(config: WorkflowConfigs) -> None:
    workflow = config.add_workflow(ExportMisfitDataJob, "EXPORT_MISFIT_DATA")
    workflow.category = "observations.correlation"

    workflow = config.add_workflow(ExportRunpathJob, "EXPORT_RUNPATH")

    workflow = config.add_workflow(DisableParametersUpdate, "DISABLE_PARAMETERS")

    workflow = config.add_workflow(MisfitPreprocessor, "MISFIT_PREPROCESSOR")
    workflow.category = "observations.correlation"

    workflow = config.add_workflow(CSVExportJob, "CSV_EXPORT")

    workflow = config.add_workflow(GenDataRFTCSVExportJob, "GEN_DATA_RFT")
