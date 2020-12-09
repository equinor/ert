from prefect import task, Flow
from prefect.engine.executors import DaskExecutor
from dask_jobqueue.lsf import LSFJob
import prefect
from ert_shared.ensemble_evaluator.entity.prefect_ensamble import get_ip_address

async def _eq_submit_job(self, script_filename):
    with open(script_filename) as fh:
        lines = fh.readlines()[1:]
    lines = [
        line.strip() if "#BSUB" not in line else line[5:].strip() for line in lines
    ]
    piped_cmd = [self.submit_command + " ".join(lines)]
    return self._call(piped_cmd, shell=True)


@task
def say_hello():
    logger = prefect.context.get("logger")
    logger.warning("Hello, Equinor!")


@task
def add(x, y=1, meta=None):
    if meta is None:
        meta = {}
    meta2 = meta.copy()
    meta2[(x, y)] = "Called :D"
    meta[(x, y)] = "Called :D"
    return {"sum": x + y, "result2": 3, "meta": meta, "meta2": meta2}


def main():
    cluster_kwargs = {
        "queue": "mr",
        "project": None,
        "cores": 1,
        "memory": "1GB",
        "use_stdin": True,
        "n_workers": 2,
        "silence_logs": "debug",
        "scheduler_options": {
            "port": 51821
        }
    }
    executor = DaskExecutor(
        cluster_class="dask_jobqueue.LSFCluster",
        cluster_kwargs=cluster_kwargs,
        debug=True,
    )

    with Flow("Test LSF Flow") as flow:
        say_hello()

        first_result = add(1, y=2)
        second_result = add(
            first_result["sum"], first_result["result2"], first_result["meta"]
        )

    state = flow.run(executor=executor)

    assert state.is_successful()

    first_task_state = state.result[first_result]

    print(first_result)
    print(second_result)
    print(state.result[first_result].result)
    print(state.result[second_result].result)
    # flow.visualize()


if __name__ == "__main__":
    LSFJob._submit_job = _eq_submit_job
    main()