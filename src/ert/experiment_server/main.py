import asyncio
import multiprocessing as mp
import uuid
from contextlib import asynccontextmanager
from multiprocessing.queues import Queue
from typing import Dict, Union

from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket
from pydantic import BaseModel, Field

from ert.config import ErtConfig
from ert.gui.simulation.ensemble_experiment_panel import (
    Arguments as EnsembleExperimentArguments,
)
from ert.gui.simulation.ensemble_smoother_panel import (
    Arguments as EnsembleSmootherArguments,
)
from ert.gui.simulation.evaluate_ensemble_panel import (
    Arguments as EvaluateEnsembleArguments,
)
from ert.gui.simulation.iterated_ensemble_smoother_panel import (
    Arguments as IteratedEnsembleSmootherArguments,
)
from ert.gui.simulation.manual_update_panel import Arguments as ManualUpdateArguments
from ert.gui.simulation.multiple_data_assimilation_panel import (
    Arguments as MultipleDataAssimilationArguments,
)
from ert.gui.simulation.single_test_run_panel import Arguments as SingleTestRunArguments
from ert.run_models.base_run_model import StatusEvents
from ert.run_models.model_factory import create_model
from ert.storage import open_storage

from .experiment_task import EndTaskEvent, ExperimentTask


class Experiment(BaseModel):
    args: Union[
        EnsembleExperimentArguments,
        EnsembleSmootherArguments,
        EvaluateEnsembleArguments,
        IteratedEnsembleSmootherArguments,
        ManualUpdateArguments,
        MultipleDataAssimilationArguments,
        SingleTestRunArguments,
    ] = Field(..., discriminator="mode")
    ert_config: ErtConfig


mp_ctx = mp.get_context("fork")
experiments: Dict[str, ExperimentTask] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    yield
    # Shutdown actions

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "ping"}


@app.post("/experiments/")
async def submit_experiment(experiment: Experiment, background_tasks: BackgroundTasks):
    storage = open_storage(experiment.ert_config.ens_path, "w")
    status_queue: "Queue[StatusEvents]" = mp_ctx.Queue()
    try:
        model = create_model(
            experiment.ert_config,
            storage,
            experiment.args,
            status_queue,
        )
    except ValueError as e:
        return HTTPException(
            status_code=420,
            detail=f"{experiment.args.mode} was not valid, failed with: {e}",
        )

    experiment_id = str(uuid.uuid4())
    task = ExperimentTask(_id=experiment_id, model=model, status_queue=status_queue)
    experiments[experiment_id] = task
    background_tasks.add_task(task.run)
    return {"message": "Experiment Started", "experiment_id": experiment_id}


@app.put("/experiments/{experiment_id}/cancel")
async def cancel_experiment(experiment_id: str):
    if experiment_id not in experiments:
        return HTTPException(
                status_code=404,
                detail=f"Experiment with id {experiment_id} does not exist.",
            )
    experiments[experiment_id].cancel()
    return {"message": "Experiment canceled", "experiment_id": experiment_id}


@app.websocket("/experiments/{experiment_id}/events")
async def websocket_endpoint(websocket: WebSocket, experiment_id: str):
    if experiment_id not in experiments:
        return
    subscriber_id = str(uuid.uuid4())
    await websocket.accept()
    task = experiments[experiment_id]
    while True:
        event = await task.get_event(subscriber_id=subscriber_id)
        if isinstance(event, EndTaskEvent):
            break
        await websocket.send_json(event)
        await asyncio.sleep(0.1)
