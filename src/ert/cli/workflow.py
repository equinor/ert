import logging


def execute_workflow(ert, workflow_name):
    logger = logging.getLogger(__name__)
    workflow_list = ert.getWorkflowList()
    try:
        workflow = workflow_list[workflow_name]
    except KeyError:
        msg = "Workflow {} is not in the list of available workflows"
        logger.error(msg.format(workflow_name))
        return
    context = workflow_list.getContext()
    workflow.run(ert=ert, verbose=True, context=context)
    all_successful = all((v["completed"] for k, v in workflow.getJobsReport().items()))
    if not all_successful:
        logger.error(f"Workflow {workflow_name} failed!")
