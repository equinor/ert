from ert.shared.plugins.plugin_manager import hook_implementation
from ert.shared.plugins.plugin_response import plugin_response

from .disable_parameters import DisableParametersUpdate
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

    # Intentionally did not add description to this workflow as we would
    # like to replace it with a better solution, see:
    # https://github.com/equinor/ert/issues/3859, if this is not done
    # use the doc string of the class as documentation
    config.add_workflow(DisableParametersUpdate, "DISABLE_PARAMETERS")
