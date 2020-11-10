from random import uniform
import subprocess
import os
import shutil
import json
import uuid
from functools import partial
from cloudevents.http import to_json, CloudEvent

import prefect
from prefect import Flow, Task
from prefect.engine.executors import DaskExecutor
from ert_shared.ensemble_evaluator.client import Client
from ert_shared.ensemble_evaluator.entity.ensemble import (
    _Ensemble, _ScriptJob, _Step, _Stage, _Realization)
from ert_shared.ensemble_evaluator.ws_util import wait_for_ws


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

    def run(self, expected_res=None):
        if expected_res is None:
            expected_res = []
        with Client(self._url) as c:
            event = _cloud_event(event_type=f"com.equinor.ert.forward_model_step.start",
                                 fm_type="step",
                                 real_id=self._iens,
                                 step_id=self._step_id,
                                 stage_id=self._stage_id)
            c.send(to_json(event))
            run_path = f"output/{self._iens}"
            os.makedirs(run_path, exist_ok=True)
            for res in expected_res:
                self._resources += res

            # Get data files needed for the run
            for data_file in self._resources:
                file_name = os.path.basename(data_file)
                sim_job_path = os.path.join(run_path, file_name)
                shutil.copyfile(data_file, sim_job_path)

            exec_metadata = {"iens": self._iens,
                             "outputs": []}
            for index, job in enumerate(self._job_list):
                print(f"Running command {self._cmd} ")
                event = _cloud_event(event_type=f"com.equinor.ert.forward_model_job.start",
                                     fm_type="job",
                                     real_id=self._iens,
                                     step_id=self._step_id,
                                     stage_id=self._stage_id,
                                     job_id=job["id"]
                                     )
                c.send(to_json(event))
                subprocess.run([self._cmd, job["executable"], *job["args"]],
                               universal_newlines=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               cwd=run_path,
                               check=True
                               )

                event = _cloud_event(event_type=f"com.equinor.ert.forward_model_job.success",
                                     fm_type="job",
                                     real_id=self._iens,
                                     step_id=self._step_id,
                                     stage_id=self._stage_id,
                                     job_id=job["id"]
                                     )
                c.send(to_json(event))
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
            event = _cloud_event(event_type=f"com.equinor.ert.forward_model_step.success",
                                 fm_type="step",
                                 real_id=self._iens,
                                 step_id=self._step_id,
                                 stage_id=self._stage_id
                                 )
            c.send(to_json(event))
        return exec_metadata


def _get_event_type(state, fm_type):
    if isinstance(state, prefect.engine.state.Pending):
        return f"com.equinor.ert.forward_model_{fm_type}.start"
    elif isinstance(state, prefect.engine.state.Running):
        return f"com.equinor.ert.forward_model_{fm_type}.running"
    elif isinstance(state, prefect.engine.state.Success):
        return f"com.equinor.ert.forward_model_{fm_type}.success"
    elif isinstance(state, prefect.engine.state.Failed):
        return f"com.equinor.ert.forward_model_{fm_type}.failure"
    else:
        return f"com.equinor.ert.forward_model_{fm_type}.unknown"


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
                                       inputs=[os.path.abspath(input_elem) for input_elem in step["resources"]],
                                       outputs=step["outputs"],
                                       jobs=jobs,
                                       name=step["name"]))
                    step["resources"] = [os.path.abspath(input_elem) for input_elem in step["resources"]]

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
                event_type=_get_event_type(state, "step"),
                fm_type="step",
                real_id=task.get_iens(),
            )
            c.send(to_json(event))

    def get_ordering(self, iens):
        table_of_elements = []
        for stage in self.config["stages"]:
            for step in stage["steps"]:
                table_of_elements.append({
                    "iens": iens,
                    "stage_name": stage["name"],
                    "stage_id": self.get_id(iens, stage_name=stage["name"]),
                    "step_id": self.get_id(iens, stage_name=stage["name"], step_name=step["name"]),
                    **step})

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

    def run_flow(self, dispatch_url, coef_input_files, real_range):
        with Flow('Test jobs') as flow:
            for iens in real_range:
                o_t_res = {}
                for step in self.get_ordering(iens=iens):
                    job_list = [{
                        "id": self.get_id(iens, step["stage_name"], step["name"], idx),
                        "name": job["name"],
                        "executable": job["executable"],
                        "args": list(map(str, job["args"])),
                    } for idx, job in enumerate(step["jobs"])]
                    inputs = [o_t_res.get(i, []) for i in step.get("inputs", [])]

                    stage_task = RunProcess(
                        resources=step.get("resources", []) + [coef_input_files[iens]],
                        outputs=step.get("outputs", []),
                        job_list=job_list,
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
        return flow.run()
        # return flow.run(executor=DaskExecutor(address="tcp://192.168.100.6:8786"))

    def evaluate(self, host, port):
        # executor = DaskExecutor(
        #     cluster_class="dask_jobqueue.LSFCluster",
        #     cluster_kwargs=cluster_kwargs,
        #     debug=True,
        # )
        dispatch_url = f"ws://{host}:{port}/dispatch"
        wait_for_ws(dispatch_url)
        coef_input_files = gen_coef(self.config["parameters"], self.config["realizations"])
        real_per_batch = self.config["max_running"]
        real_range = range(self.config["realizations"])
        i = 0
        while i < self.config["realizations"]:
            self.run_flow(dispatch_url, coef_input_files, real_range[i:i+real_per_batch])
            i = i + real_per_batch

