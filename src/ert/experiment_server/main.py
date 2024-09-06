import queue

from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket
from pydantic import BaseModel, Field
import json
import dataclasses
import asyncio

from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.run_models.model_factory import create_model
from ert.run_models.base_run_model import BaseRunModel, StatusEvents
from ert.gui.simulation.ensemble_experiment_panel import Arguments as EnsembleExperimentArguments
from ert.gui.simulation.ensemble_smoother_panel import Arguments as EnsembleSmootherArguments
from ert.gui.simulation.evaluate_ensemble_panel import Arguments as EvaluateEnsembleArguments
from ert.gui.simulation.iterated_ensemble_smoother_panel import Arguments as IteratedEnsembleSmootherArguments
from ert.gui.simulation.manual_update_panel import Arguments as ManualUpdateArguments
from ert.gui.simulation.multiple_data_assimilation_panel import Arguments as MultipleDataAssimilationArguments
from ert.gui.simulation.single_test_run_panel import Arguments as SingleTestRunArguments
from ert.storage import open_storage

from typing import Dict, Union, Tuple
import uuid

from ert.config import ErtConfig, QueueSystem
from fastapi.encoders import jsonable_encoder

class Experiment(BaseModel):
    args: Union[EnsembleExperimentArguments, EnsembleSmootherArguments, EvaluateEnsembleArguments, IteratedEnsembleSmootherArguments, ManualUpdateArguments, MultipleDataAssimilationArguments, SingleTestRunArguments] = Field(..., discriminator='mode')
    ert_config: ErtConfig

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "ping"}

experiments : Dict[str, Tuple[BaseRunModel, queue.SimpleQueue]]= {}

def run_experiment(experiment_id:str, evaluator_server_config: EvaluatorServerConfig):
    experiments[experiment_id][0].start_simulations_thread(evaluator_server_config=evaluator_server_config)

@app.post("/experiments/")
async def submit_experiment(experiment: Experiment, background_tasks: BackgroundTasks):
    storage = open_storage(experiment.ert_config.ens_path, "w")
    status_queue: queue.SimpleQueue[StatusEvents] = queue.SimpleQueue()
    try:
        model = create_model(
            experiment.ert_config,
            storage,
            experiment.args,
            status_queue,
        )
    except ValueError as e:
        return HTTPException(status_code=404, detail=f"{experiment.args.mode} was not valid, failed with: {e}")

    port_range = None
    if model.queue_system == QueueSystem.LOCAL:
        port_range = range(49152, 51819)
    evaluator_server_config = EvaluatorServerConfig(custom_port_range=port_range)

    experiment_id = str(uuid.uuid4())
    experiments[experiment_id] = (model, status_queue)

    background_tasks.add_task(run_experiment, experiment_id, evaluator_server_config=evaluator_server_config)
    return {"message": "Experiment Started", "experiment_id": experiment_id}

@app.put("/experiments/{experiment_id}/cancel")
async def cancel_experiment(experiment_id: str):
    if experiment_id in experiments:
        experiments[experiment_id][0].cancel()
    return {"message": "Experiment Canceled", "experiment_id": experiment_id}


@app.websocket("/experiments/{experiment_id}/events")
async def websocket_endpoint(websocket: WebSocket, experiment_id: str):
    await websocket.accept()
    print(experiment_id)
    print(experiments)
    q = experiments[experiment_id][1]
    while True:
        item: StatusEvents = q.get()
        from ert.ensemble_evaluator.event import _UpdateEvent, EndEvent
        if isinstance(item, _UpdateEvent):
            item.snapshot = item.snapshot.to_dict()
        print(item)
        print()
        print()
        await websocket.send_json(jsonable_encoder(item))
        await asyncio.sleep(0.1)
        if isinstance(item, EndEvent):
            break
        
