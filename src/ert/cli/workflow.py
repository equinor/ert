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
    failed_jobs = workflow.get_failed_jobs()
    if failed_jobs:
        failed_jobs_str = ",".join(failed_jobs)
        logger.error(
            f"Workflow {workflow_name} Failed! Jobs: [{failed_jobs_str}] Failed!"
        )
