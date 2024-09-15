import asyncio
import multiprocessing as mp
import uuid
from multiprocessing.queues import Queue
from typing import Dict, List

from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket

from ert.run_models.base_run_model import StatusEvents
from ert.run_models.model_factory import create_model
from ert.storage import open_storage

from .experiment_task import EndTaskEvent, ExperimentTask
from .models import Experiment, ExperimentOut

experiments: Dict[str, ExperimentTask] = {}
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "ping"}


@app.get("/experiments/", response_model=List[ExperimentOut])
async def get_experiments():
    return [ExperimentOut(id=k, type=v.model_type) for k, v in experiments.items()]


@app.post("/experiments/", response_model=ExperimentOut)
async def submit_experiment(experiment: Experiment, background_tasks: BackgroundTasks):
    storage = open_storage(experiment.ert_config.ens_path, "w")
    status_queue: "Queue[StatusEvents]" = mp.Queue()
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
    return ExperimentOut(id=experiment_id, type=task.model_type)


@app.put("/experiments/{experiment_id}/cancel", response_model=ExperimentOut)
async def cancel_experiment(experiment_id: str):
    if experiment_id not in experiments:
        return HTTPException(
                status_code=404,
                detail=f"Experiment with id {experiment_id} does not exist.",
            )
    task = experiments[experiment_id]
    task.cancel()
    return ExperimentOut(id=experiment_id, type=task.model_type)


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
