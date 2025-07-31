import hashlib
import json
import os
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _ert.forward_model_runner.fm_dispatch import ForwardModelDescriptionJSON
from _ert.forward_model_runner.forward_model_step import ForwardModelStep
from _ert.forward_model_runner.reporting.message import (
    Checksum,
    Finish,
    Init,
    Manifest,
    Message,
)


class ForwardModelRunner:
    def __init__(self, steps_data: "ForwardModelDescriptionJSON") -> None:
        self.steps_data = (
            steps_data  # On disk, this is called jobs.json for legacy reasons
        )
        self.simulation_id = steps_data.get("run_id")
        self.experiment_id = steps_data.get("experiment_id")
        self.ens_id = steps_data.get("ens_id")
        self.real_id = steps_data.get("real_id")
        self.ert_pid = steps_data.get("ert_pid")
        self.global_environment = steps_data.get("global_environment")
        if self.simulation_id is not None:
            os.environ["ERT_RUN_ID"] = self.simulation_id

        self.steps: list[ForwardModelStep] = []
        for index, step_data in enumerate(steps_data["jobList"]):
            self.steps.append(ForwardModelStep(step_data, index))
        self._currently_running_step: ForwardModelStep | None = None
        self._set_environment()

    def _read_manifest(self) -> dict[str, Manifest] | None:
        if not Path("manifest.json").exists():
            return None
        with open("manifest.json", encoding="utf-8") as f:
            data = json.load(f)
        return {
            name: {"type": "file", "path": str(Path(file).absolute())}
            for name, file in data.items()
        }

    def _populate_checksums(
        self, manifest: dict[str, Manifest] | None
    ) -> dict[str, Manifest]:
        if not manifest:
            return {}
        for info in manifest.values():
            path = Path(info["path"])
            if path.exists():
                info["md5sum"] = hashlib.md5(path.read_bytes()).hexdigest()
            else:
                info["error"] = f"Expected file {path} not created by forward model!"
        return manifest

    def run(self, names_of_steps_to_run: list[str]) -> Generator[Message]:
        if not names_of_steps_to_run:
            step_queue = self.steps
        else:
            step_queue = [
                step for step in self.steps if step.name() in names_of_steps_to_run
            ]
        init_message = Init(
            step_queue,
            self.simulation_id,
            self.ert_pid,
            self.ens_id,
            self.real_id,
            self.experiment_id,
        )

        unused = set(names_of_steps_to_run) - {step.name() for step in step_queue}
        if unused:
            init_message.with_error(
                f"{unused} does not exist. "
                f"Available forward_model steps: {[step.name() for step in self.steps]}"
            )
            yield init_message
            yield Finish().with_error(
                "Not all forward model steps completed successfully."
            )
            return
        else:
            yield init_message

        for step in step_queue:
            self._currently_running_step = step
            for status_update in step.run():
                yield status_update
                if not status_update.success():
                    yield Checksum(checksum_dict={}, run_path=os.getcwd())
                    yield Finish().with_error(
                        "Not all forward model steps completed successfully."
                    )
                    return

        checksum_dict = self._populate_checksums(self._read_manifest())
        yield Checksum(checksum_dict=checksum_dict, run_path=os.getcwd())
        yield Finish()

    def _set_environment(self) -> None:
        if self.global_environment:
            for key, value in self.global_environment.items():
                for env_key, env_val in os.environ.items():
                    value = value.replace(f"${env_key}", env_val)
                os.environ[key] = value
