import hashlib
import json
import os
from pathlib import Path

from _ert_forward_model_runner.job import Job
from _ert_forward_model_runner.reporting.message import Checksum, Finish, Init


class ForwardModelRunner:
    def __init__(self, jobs_data):
        self.jobs_data = jobs_data
        self.simulation_id = jobs_data.get("run_id")
        self.experiment_id = jobs_data.get("experiment_id")
        self.ens_id = jobs_data.get("ens_id")
        self.real_id = jobs_data.get("real_id")
        self.ert_pid = jobs_data.get("ert_pid")
        self.global_environment = jobs_data.get("global_environment")
        job_data_list = jobs_data["jobList"]

        if self.simulation_id is not None:
            os.environ["ERT_RUN_ID"] = self.simulation_id

        self.jobs = []
        for index, job_data in enumerate(job_data_list):
            self.jobs.append(Job(job_data, index))

        self._set_environment()

    def _read_manifest(self):
        if not Path("manifest.json").exists():
            return None
        with open("manifest.json", mode="r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            name: {"type": "file", "path": str(Path(file).absolute())}
            for name, file in data.items()
        }

    def _populate_checksums(self, manifest):
        if not manifest:
            return None
        for info in manifest.values():
            path = Path(info["path"])
            if path.exists():
                info["md5sum"] = hashlib.md5(path.read_bytes()).hexdigest()
            else:
                info["error"] = f"Expected file {path} not created by forward model!"
        return manifest

    def run(self, names_of_jobs_to_run):
        # if names_of_jobs_to_run, create job_queue which contains jobs that
        # are to be run.
        if not names_of_jobs_to_run:
            job_queue = self.jobs
        else:
            job_queue = [j for j in self.jobs if j.name() in names_of_jobs_to_run]
        init_message = Init(
            job_queue,
            self.simulation_id,
            self.ert_pid,
            self.ens_id,
            self.real_id,
            self.experiment_id,
        )

        unused = set(names_of_jobs_to_run) - {j.name() for j in job_queue}
        if unused:
            init_message.with_error(
                f"{unused} does not exist. "
                f"Available jobs: {[j.name() for j in self.jobs]}"
            )
            yield init_message
            return
        else:
            yield init_message

        for job in job_queue:
            for status_update in job.run():
                yield status_update

                if not status_update.success():
                    yield Checksum(checksum_dict=None, run_path=os.getcwd())
                    yield Finish().with_error("Not all jobs completed successfully.")
                    return

        checksum_dict = self._populate_checksums(self._read_manifest())
        yield Checksum(checksum_dict=checksum_dict, run_path=os.getcwd())
        yield Finish()

    def _set_environment(self):
        if self.global_environment:
            for key, value in self.global_environment.items():
                for env_key, env_val in os.environ.items():
                    value = value.replace(f"${env_key}", env_val)
                os.environ[key] = value
