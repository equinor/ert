from ert_shared.plugins.plugin_manager import hook_implementation
from ert_shared.plugins.plugin_response import plugin_response

from .export_misfit_data import ExportMisfitDataJob
from .export_runpath import ExportRunpathJob


@hook_implementation
@plugin_response(plugin_name="ert")
def legacy_ertscript_workflow(config):
    workflow = config.add_workflow(ExportMisfitDataJob, "EXPORT_MISFIT_DATA")
    workflow.description = ExportMisfitDataJob.__doc__
    workflow.category = "observations.correlation"

    workflow = config.add_workflow(ExportRunpathJob, "EXPORT_RUNPATH")
    workflow.description = ExportRunpathJob.__doc__
