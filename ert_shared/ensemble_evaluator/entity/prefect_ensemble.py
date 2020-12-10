from random import uniform
import subprocess
import os
import shutil
import json
import uuid
import threading
from functools import partial
from cloudevents.http import to_json, CloudEvent
from prefect import Flow, Task
from prefect.engine.executors import DaskExecutor
from dask_jobqueue.lsf import LSFJob
from ert_shared.ensemble_evaluator.entity import identifiers as ids
from ert_shared.ensemble_evaluator.client import Client
from ert_shared.ensemble_evaluator.config import find_open_port
from ert_shared.ensemble_evaluator.entity.ensemble import (
    _Ensemble, _BaseJob, _Step, _Stage, _Realization)


async def _eq_submit_job(self, script_filename):
    with open(script_filename) as fh:
        lines = fh.readlines()[1:]
    lines = [
        line.strip() if "#BSUB" not in line else line[5:].strip() for line in lines
    ]
    piped_cmd = [self.submit_command + " ".join(lines)]
    return self._call(piped_cmd, shell=True)


def _get_executor(name="local"):
    if name == "local":
        return None
    elif name == "lsf":
        LSFJob._submit_job = _eq_submit_job
        cluster_kwargs = {
            "queue": "mr",
            "project": None,
            "cores": 1,
            "memory": "1GB",
            "use_stdin": True,
            "n_workers": 2,
            "silence_logs": "debug",
            "scheduler_options": {
                "port": find_open_port(lower=51820, upper=51840)
            },
        }
        return DaskExecutor(
            cluster_class="dask_jobqueue.LSFCluster",
            cluster_kwargs=cluster_kwargs,
            debug=True,
        )
    elif name == "pbs":
        cluster_kwargs = {
            "n_workers": 10,
            "queue": "normal",
            "project": "ERT-TEST",
            "local_directory": "$TMPDIR",
            "cores": 4,
            "memory": "16GB",
            "resource_spec": "select=1:ncpus=4:mem=16GB",
        }
        return DaskExecutor(
            cluster_class="dask_jobqueue.PBSCluster",
            cluster_kwargs=cluster_kwargs,
            debug=True,
        )
    else:
        raise ValueError(f"Unknown executor name {name}")


def storage_driver_factory(config, run_path):
    if config.get("storage"):
        if config["storage"].get("type") == "shared_disk":
            storage_path = config["storage"]["storage_path"]
            return SharedDiskStorageDriver(storage_path, run_path)
        else:
            raise ValueError(f"Not a valid storage type. ({config['storage'].get('type')})")
    else:
        # default
        storage_path = os.path.join(os.getcwd(), "storage_folder")
        return SharedDiskStorageDriver(storage_path, run_path)


class SharedDiskStorageDriver:
    def __init__(self, storage_path, run_path):
        self._storage_path = storage_path
        self._run_path = f"{run_path}"

    def get_storage_path(self, iens):
        if iens is None:
            return f"{self._storage_path}/global"
        return f"{self._storage_path}/{iens}"

    def store(self, local_name, iens=None):
        storage_path = self.get_storage_path(iens)
        os.makedirs(storage_path, exist_ok=True)
        storage_uri = os.path.join(storage_path, local_name)
        shutil.copyfile(os.path.join(self._run_path, local_name), storage_uri)
        return storage_uri

    def retrieve(self, storage_uri):
        if storage_uri.startswith(self._storage_path):
            target = os.path.basename(storage_uri)
            shutil.copyfile(storage_uri, os.path.join(self._run_path, target))
            return target
        else:
            raise ValueError(f"Storage driver can't handle file: {storage_uri}")


class RunProcess(Task):
    def __init__(self, resources, outputs, job_list, iens, cmd, url,
                 step_id, stage_id, runtime_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._resources = resources
        self._outputs = outputs
        self._job_list = job_list
        self._iens = iens
        self._cmd = cmd
        self._url = url
        self._step_id = step_id
        self._stage_id = stage_id
        self._runtime_config = runtime_config

    def get_iens(self):
        return self._iens

    def get_stage_id(self):
        return self._stage_id

    def get_step_id(self):
        return self._step_id

    def run(self, expected_res=None):
        if expected_res is None:
            expected_res = []
        else:
            expected_res = [item for sublist in expected_res for item in sublist]
        with Client(self._url) as c:
            event = _cloud_event(event_type=ids.EVTYPE_FM_STEP_START,
                                 fm_type="step",
                                 real_id=self._iens,
                                 step_id=self._step_id,
                                 stage_id=self._stage_id)
            c.send(to_json(event).decode())
            run_path = os.path.join(self._runtime_config["run_path"], str(self._iens))

            storage = storage_driver_factory(self._runtime_config, run_path)
            os.makedirs(run_path, exist_ok=True)

            expected_res += self._resources
            for r in expected_res:
                storage.retrieve(r)

            exec_metadata = {"iens": self._iens,
                             "outputs": []}
            for index, job in enumerate(self._job_list):
                print(f"Running command {self._cmd}  {job['name']}")
                event = _cloud_event(event_type=ids.EVTYPE_FM_JOB_START,
                                     fm_type="job",
                                     real_id=self._iens,
                                     step_id=self._step_id,
                                     stage_id=self._stage_id,
                                     job_id=job["id"]
                                     )
                c.send(to_json(event).decode())
                cmd_exec = subprocess.run([self._cmd, job["executable"], *job["args"]],
                                          universal_newlines=True, stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE,
                                          cwd=run_path,
                                          )
                self.logger.info(cmd_exec.stdout)
                if cmd_exec.returncode != 0:
                    self.logger.error(cmd_exec.stderr)
                    event = _cloud_event(event_type=ids.EVTYPE_FM_JOB_FAILURE,
                                         fm_type="job",
                                         real_id=self._iens,
                                         step_id=self._step_id,
                                         stage_id=self._stage_id,
                                         job_id=job["id"],
                                         data={"stderr": cmd_exec.stderr,
                                               "stdout": cmd_exec.stdout}
                                         )
                    c.send(to_json(event).decode())
                    raise RuntimeError(f"Script {job['name']} failed with exception {cmd_exec.stderr}")

                event = _cloud_event(event_type=ids.EVTYPE_FM_JOB_SUCCESS,
                                     fm_type="job",
                                     real_id=self._iens,
                                     step_id=self._step_id,
                                     stage_id=self._stage_id,
                                     job_id=job["id"],
                                     data={"stdout": cmd_exec.stdout}
                                     )
                c.send(to_json(event).decode())

            for output in self._outputs:
                if not os.path.exists(os.path.join(run_path, output)):
                    exec_metadata["error"] = f"Output file {output} was not generated!"

                storage.store(output, self._iens)

            event = _cloud_event(event_type=ids.EVTYPE_FM_STEP_SUCCESS,
                                 fm_type="step",
                                 real_id=self._iens,
                                 step_id=self._step_id,
                                 stage_id=self._stage_id
                                 )
            c.send(to_json(event).decode())
        return exec_metadata


def _build_source(fm_type, real_id=None, stage_id=None,  step_id=None, job_id=None):
    source_map = {
        "stage": f"/ert/ee/0/real/{real_id}/stage/{stage_id}",
        "step": f"/ert/ee/0/real/{real_id}/stage/{stage_id}/step/{step_id}",
        "job": f"/ert/ee/0/real/{real_id}/stage/{stage_id}/step/{step_id}/job/{job_id}",
        "ensemble": "/ert/ee/0"
    }
    return source_map.get(fm_type, f"/ert/ee/0")


def _cloud_event(event_type, fm_type, real_id=None, stage_id=None, step_id=None, job_id=None, data=None):
    if data is None:
        data = {}
    return CloudEvent(
        {
            "type": event_type,
            "source": _build_source(fm_type, real_id, stage_id, step_id, job_id),
            "datacontenttype": "application/json",
        },
        data
    )


def gen_coef(parameters, real, config):
    data = {}
    paths = {}
    storage = storage_driver_factory(config, "coeff")
    for iens in range(real):
        for name, elements in parameters.items():
            for element in elements:
                start, end = element["args"]
                data[element["name"]] = uniform(start, end)
        os.makedirs(f"coeff", exist_ok=True)
        file_name = f"coeffs.json"
        file_path = os.path.join("coeff", file_name)
        with open(file_path, "w") as f:
            json.dump(data, f)
        paths[iens] = storage.store(file_name, iens)
    return paths


class PrefectEnsemble(_Ensemble):
    def __init__(self, config):
        self.config = config
        self._reals = self._get_reals()
        super().__init__(self._reals, metadata={"iter": 0})

    def _get_reals(self):
        reals = []
        for iens in range(self.config['realizations']):
            stages = []
            for stage in self.config['stages']:
                steps = []
                stage_id = uuid.uuid4()
                for step in stage["steps"]:
                    jobs = []
                    for job in step["jobs"]:
                        job_id = uuid.uuid4()
                        jobs.append(_BaseJob(id_=str(job_id), name=job["name"]))
                    step_id = uuid.uuid4()
                    steps.append(_Step(id_=str(step_id),
                                       inputs=step.get("inputs", []),
                                       outputs=step["outputs"],
                                       jobs=jobs,
                                       name=step["name"]))

                stages.append(_Stage(id_=str(stage_id),
                                     steps=steps,
                                     status="Unknown",
                                     name=stage["name"]))
            reals.append(_Realization(iens=iens, stages=stages, active=True))
        return reals

    @staticmethod
    def _on_task_failure(task, state, url):
        with Client(url) as c:
            event = _cloud_event(
                event_type=ids.EVTYPE_FM_STEP_FAILURE,
                fm_type="step",
                real_id=task.get_iens(),
                stage_id=task.get_stage_id(),
                step_id=task.get_step_id(),
                data={"task_state": state.message}
            )
            c.send(to_json(event).decode())

    def get_id(self, iens, stage_name, step_name=None, job_index=None):
        real = next(x for x in self._reals if x.get_iens() == iens)
        stage = next(x for x in real.get_stages() if x.get_name() == stage_name)
        if step_name is None:
            return stage.get_id()
        step = next(x for x in stage.get_steps() if x.get_name() == step_name)
        if job_index is None:
            return step.get_id()
        job = step.get_jobs()[job_index]
        return job.get_id()

    def get_ordering(self, iens):
        table_of_elements = []
        for stage in self.config["stages"]:
            for step in stage["steps"]:
                jobs = [{
                    "id": self.get_id(iens, stage_name=stage["name"], step_name=step["name"], job_index=idx),
                    "name": job.get("name"),
                    "executable": job.get("executable"),
                    "args": job.get("args", []),
                } for idx, job in enumerate(step.get("jobs", []))]
                table_of_elements.append({
                    "iens": iens,
                    "stage_name": stage["name"],
                    "stage_id": self.get_id(iens, stage_name=stage["name"]),
                    "step_id": self.get_id(iens, stage_name=stage["name"], step_name=step["name"]),
                    **step,
                    "jobs": jobs,
                })

        produced = set()
        ordering = []
        while table_of_elements:
            temp_list = produced.copy()
            for element in table_of_elements:
                if set(element.get("inputs", [])).issubset(temp_list):
                    ordering.append(element)
                    produced = produced.union(set(element["outputs"]))
                    table_of_elements.remove(element)
        return ordering

    def run_flow(self, dispatch_url, coef_input_files, real_range, executor):
        with Flow(f"Realization range {real_range}") as flow:
            for iens in real_range:
                o_t_res = {}
                for step in self.get_ordering(iens=iens):
                    inputs = [o_t_res.get(i, []) for i in step.get("inputs", [])]
                    stage_task = RunProcess(
                        resources=[coef_input_files[iens]] + step["resources"],
                        outputs=step.get("outputs", []),
                        job_list=step.get("jobs", []),
                        iens=iens,
                        cmd="python3",
                        url=dispatch_url,
                        step_id=step["step_id"],
                        stage_id=step["stage_id"],
                        on_failure=partial(self._on_task_failure, url=dispatch_url),
                        runtime_config=self.config
                    )
                    result = stage_task(expected_res=inputs)

                    for o in step.get("outputs", []):
                        o_t_res[o] = result["outputs"]
        return flow.run(executor=executor)

    def evaluate(self, config, ee_id):
        print(f"Running with executor {self.config['executor'].upper()}")
        evaluate_thread = threading.Thread(target=self._evaluate, args=(config, ee_id))
        evaluate_thread.start()

    def _evaluate(self, config, ee_id):
        dispatch_url = f"ws://{config.get('host')}:{config.get('port')}/dispatch"
        coef_input_files = gen_coef(self.config["parameters"], self.config["realizations"], self.config)

        real_per_batch = self.config["max_running"]
        real_range = range(self.config["realizations"])

        storage = storage_driver_factory(self.config, self.config.get("config_path", "."))
        for stage in self.config["stages"]:
            for step in stage["steps"]:
                res_list = []
                for res in step["resources"]:
                    res_list.append(storage.store(res))
                step["resources"] = res_list

        i = 0
        executor = _get_executor(self.config["executor"])
        while i < self.config["realizations"]:
            self.run_flow(dispatch_url, coef_input_files, real_range[i:i+real_per_batch], executor)
            i = i + real_per_batch

        with Client(dispatch_url) as c:
            event = _cloud_event(
                event_type=ids.EVTYPE_ENSEMBLE_STOPPED,
                fm_type="ensemble"
            )
            c.send(to_json(event).decode())

