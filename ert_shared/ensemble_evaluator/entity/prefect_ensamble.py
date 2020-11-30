from random import uniform
import subprocess
import os
import shutil
import json
import uuid
from functools import partial
from cloudevents.http import to_json, CloudEvent
from prefect import Flow, Task
from prefect.engine.executors import DaskExecutor

from ert_shared.ensemble_evaluator.entity import identifiers as ids
from ert_shared.ensemble_evaluator.client import Client
from ert_shared.ensemble_evaluator.entity.ensemble import (
    _Ensemble, _ScriptJob, _Step, _Stage, _Realization)


class RunProcess(Task):
    def __init__(self, resources, outputs, job_list, iens, cmd, url,
                 step_id, stage_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._resources = resources
        self._outputs = outputs
        self._job_list = job_list
        self._iens = iens
        self._cmd = cmd
        self._url = url
        self._step_id = step_id
        self._stage_id = stage_id

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
            run_path = f"output/{self._iens}"
            os.makedirs(run_path, exist_ok=True)

            expected_res += self._resources
            expected_res += [job["executable"] for job in self._job_list]

            # Get data files needed for the run
            for data_file in expected_res:
                file_name = os.path.basename(data_file)
                sim_job_path = os.path.join(run_path, file_name)
                shutil.copyfile(data_file, sim_job_path)

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
                    raise RuntimeError(f"Script {job['name']} filed with exception {cmd_exec.stderr}")

                event = _cloud_event(event_type=ids.EVTYPE_FM_JOB_SUCCESS,
                                     fm_type="job",
                                     real_id=self._iens,
                                     step_id=self._step_id,
                                     stage_id=self._stage_id,
                                     job_id=job["id"],
                                     data={"stdout": cmd_exec.stdout}
                                     )
                c.send(to_json(event).decode())
                exec_metadata[job["name"]] = {}

            for output in self._outputs:
                if not os.path.exists(os.path.join(run_path, output)):
                    exec_metadata["error"] = f"Output file {output} was not generated!"

                # Storage happens here
                output_path = f"/home/dan/.prefect/{run_path}"
                os.makedirs(output_path, exist_ok=True)

                shutil.copyfile(os.path.join(run_path, output), os.path.join(output_path, output))
                exec_metadata["outputs"].append(os.path.join(output_path, output))

            exec_metadata["cwd"] = os.getcwd()
            event = _cloud_event(event_type=ids.EVTYPE_FM_STEP_SUCCESS,
                                 fm_type="step",
                                 real_id=self._iens,
                                 step_id=self._step_id,
                                 stage_id=self._stage_id
                                 )
            c.send(to_json(event).decode())
        return exec_metadata


def _build_source(fm_type, real_id, stage_id=None,  step_id=None, job_id=None):
    source_map = {
        "stage": f"/ert/ee/0/real/{real_id}/stage/{stage_id}",
        "step": f"/ert/ee/0/real/{real_id}/stage/{stage_id}/step/{step_id}",
        "job": f"/ert/ee/0/real/{real_id}/stage/{stage_id}/step/{step_id}/job/{job_id}",
    }
    return source_map.get(fm_type, f"/ert/ee/0")


def _cloud_event(event_type, fm_type, real_id, stage_id=None, step_id=None, job_id=None, data=None):
    if data is None:
        data = {}
    if fm_type == "stage":
        data.update({"queue_event_type": ""})

    return CloudEvent(
        {
            "type": event_type,
            "source": _build_source(fm_type, real_id, stage_id, step_id, job_id),
            "datacontenttype": "application/json",
        },
        data
    )


def gen_coef(parameters, real):
    data = {}
    paths = {}
    for iens in range(real):
        for name, elements in parameters.items():
            for element in elements:
                start, end = element["args"]
                data[element["name"]] = uniform(start, end)
        os.makedirs(f"coeff/{iens}", exist_ok=True)
        file_name = f"coeff/{iens}/coeffs.json"
        with open(file_name, "w") as f:
            json.dump(data, f)
        paths[iens] = os.path.abspath(file_name)
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
                        jobs.append(_ScriptJob(id_=str(job_id),
                                               name=job["name"],
                                               executable=job["executable"],
                                               args=tuple(map(str, job["args"]))))
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
                step_id=task.get_step_id()
            )
            c.send(to_json(event).decode())

    def get_ordering(self, iens):
        realization = next(real for real in self.get_reals() if real.get_iens() == iens)
        table_of_elements = []
        for stage in realization.get_stages():
            for step in stage.get_steps():
                jobs = [{
                    "id": job.get_id(),
                    "name": job.get_name(),
                    "executable": os.path.abspath(job.get_executable()),
                    "args": job.get_args(),
                } for job in step.get_jobs()]
                table_of_elements.append({
                    "iens": iens,
                    "stage_name": stage.get_name(),
                    "stage_id": stage.get_id(),
                    "step_id": step.get_id(),
                    "inputs": step.get_inputs(),
                    "outputs": step.get_outputs(),
                    "jobs": jobs})

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

    def run_flow(self, dispatch_url, coef_input_files, real_range):
        with Flow(f"Realization range {real_range}") as flow:
            for iens in real_range:
                o_t_res = {}
                for step in self.get_ordering(iens=iens):
                    inputs = [o_t_res.get(i, []) for i in step.get("inputs", [])]
                    stage_task = RunProcess(
                        resources=[coef_input_files[iens]],
                        outputs=step.get("outputs", []),
                        job_list=step.get("jobs", []),
                        iens=iens,
                        cmd="python",
                        url=dispatch_url,
                        step_id=step["step_id"],
                        stage_id=step["stage_id"],
                        on_failure=partial(self._on_task_failure, url=dispatch_url)
                    )
                    result = stage_task(expected_res=inputs)

                    for o in step.get("outputs", []):
                        o_t_res[o] = result["outputs"]
        # return flow.run()
        return flow.run(executor=DaskExecutor(address="tcp://192.168.100.6:8786"))

    def evaluate(self, host, port):
        # executor = DaskExecutor(
        #     cluster_class="dask_jobqueue.LSFCluster",
        #     cluster_kwargs=cluster_kwargs,
        #     debug=True,
        # )
        dispatch_url = f"ws://{host}:{port}/dispatch"
        coef_input_files = gen_coef(self.config["parameters"], self.config["realizations"])
        real_per_batch = self.config["max_running"]
        real_range = range(self.config["realizations"])
        i = 0
        while i < self.config["realizations"]:
            self.run_flow(dispatch_url, coef_input_files, real_range[i:i+real_per_batch])
            i = i + real_per_batch

