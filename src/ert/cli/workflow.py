import logging

from ert._c_wrappers.job_queue import WorkflowRunner


def execute_workflow(ert, storage, workflow_name):
    logger = logging.getLogger(__name__)
    try:
        workflow = ert.resConfig().workflows[workflow_name]
    except KeyError:
        msg = "Workflow {} is not in the list of available workflows"
        logger.error(msg.format(workflow_name))
        return
    runner = WorkflowRunner(workflow, ert, storage)
    runner.run_blocking()
    if not all(v["completed"] for v in runner.workflowReport().values()):
        logger.error(f"Workflow {workflow_name} failed!")
