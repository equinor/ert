import ert

from .csv_export2 import CsvExport2Job
from .disable_parameters import DisableParametersUpdate
from .export_misfit_data import ExportMisfitDataJob
from .export_runpath import ExportRunpathJob
from .misfit_preprocessor import MisfitPreprocessor


@ert.plugin(name="ert")  # type: ignore
def legacy_ertscript_workflow(config):
    workflow = config.add_workflow(ExportMisfitDataJob, "EXPORT_MISFIT_DATA")
    workflow.description = ExportMisfitDataJob.__doc__
    workflow.category = "observations.correlation"

    workflow = config.add_workflow(ExportRunpathJob, "EXPORT_RUNPATH")
    workflow.description = ExportRunpathJob.__doc__

    workflow = config.add_workflow(CsvExport2Job, "CSV_EXPORT2")
    workflow.parser = CsvExport2Job.parser
    workflow.description = CsvExport2Job.description
    workflow.examples = CsvExport2Job.examples
    workflow.category = "export"

    workflow = config.add_workflow(DisableParametersUpdate, "DISABLE_PARAMETERS")
    workflow.description = DisableParametersUpdate.__doc__

    workflow = config.add_workflow(MisfitPreprocessor, "MISFIT_PREPROCESSOR")
    workflow.description = MisfitPreprocessor.__doc__
    workflow.category = "observations.correlation"
