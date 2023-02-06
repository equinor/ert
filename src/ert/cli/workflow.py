import logging


def execute_workflow(ert, workflow_name):
    logger = logging.getLogger(__name__)
    try:
        workflow = ert.resConfig().workflows[workflow_name]
    except KeyError:
        msg = "Workflow {} is not in the list of available workflows"
        logger.error(msg.format(workflow_name))
        return
    workflow.run(ert=ert)
    if not all(v["completed"] for v in workflow.getJobsReport().values()):
        logger.error(f"Workflow {workflow_name} failed!")
