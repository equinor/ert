import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from _ert.forward_model_runner.forward_model_step import ForwardModelStep
from _ert.forward_model_runner.reporting.message import (
    Checksum,
    Finish,
    Init,
    Message,
)


class ForwardModelRunner:
    def __init__(
        self, steps_data: Dict[str, Any], reporter_queue: asyncio.Queue[Message]
    ):
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

        self.steps: List[ForwardModelStep] = []
        for index, step_data in enumerate(steps_data["jobList"]):
            self.steps.append(ForwardModelStep(step_data, index))
        self._reporter_queue = reporter_queue
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
            return {}
        for info in manifest.values():
            path = Path(info["path"])
            if path.exists():
                info["md5sum"] = hashlib.md5(path.read_bytes()).hexdigest()
            else:
                info["error"] = f"Expected file {path} not created by forward model!"
        return manifest

    async def run(self, names_of_steps_to_run: List[str]) -> None:
        try:
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
                await self.put_event(init_message)
                return

            await self.put_event(init_message)
            for step in step_queue:
                async for status_update in step.run():
                    await self.put_event(status_update)
                    if not status_update.success():
                        await self.put_event(
                            Checksum(checksum_dict={}, run_path=os.getcwd())
                        )
                        await self.put_event(
                            Finish().with_error(
                                f"Not all forward model steps completed successfully ({status_update.error_message})."
                            )
                        )
                        return
            checksum_dict = self._populate_checksums(self._read_manifest())
            await self.put_event(
                Checksum(checksum_dict=checksum_dict, run_path=os.getcwd())
            )
            await self.put_event(Finish())
            return
        except asyncio.CancelledError:
            await self.put_event(Checksum(checksum_dict={}, run_path=os.getcwd()))
            await self.put_event(
                Finish().with_error(
                    "Not all forward model steps completed successfully."
                )
            )
            return

    def _set_environment(self):
        if self.global_environment:
            for key, value in self.global_environment.items():
                for env_key, env_val in os.environ.items():
                    value = value.replace(f"${env_key}", env_val)
                os.environ[key] = value

    async def put_event(self, event: Message):
        await self._reporter_queue.put(event)
