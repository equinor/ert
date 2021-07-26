import asyncio
import pathlib
import pickle
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cloudpickle
from pydantic import FilePath
import ert
import ert3
from ert3.config import EnsembleConfig, StagesConfig, Step
from ert.data import RecordCollection, RecordCollectionMap, Record, RecordTransmitter
from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig
from ert_shared.ensemble_evaluator.entity.identifiers import EVTYPE_EE_TERMINATED
from ert_shared.ensemble_evaluator.evaluator import EnsembleEvaluator
from ert_shared.ensemble_evaluator.ensemble.prefect import PrefectEnsemble

_EVTYPE_SNAPSHOT_STOPPED = "Stopped"
_EVTYPE_SNAPSHOT_FAILED = "Failed"


def _create_evaluator_tmp_dir(workspace_root: Path, evaluation_name: str) -> Path:
    return (
        pathlib.Path(workspace_root)
        / ert3._WORKSPACE_DATA_ROOT
        / "tmp"
        / evaluation_name
    )


def _prepare_input(
    storage_type: str,
    step_config: Step,
    inputs: RecordCollectionMap,
    storage_path: str,
    ensemble_size: int,
) -> Dict[int, Dict[str, RecordTransmitter]]:
    transmitters: Dict[int, Dict[str, ert.data.RecordTransmitter]] = defaultdict(dict)

    futures = []
    for input_ in step_config.input:
        for iens, record in enumerate(inputs.ensemble_records[input_.record].records):
            transmitter: RecordTransmitter
            if storage_type == "shared_disk":
                transmitter = ert.data.SharedDiskRecordTransmitter(
                    name=input_.record,
                    storage_path=pathlib.Path(storage_path),
                )
            elif storage_type == "ert_storage":
                transmitter = ert.storage.StorageRecordTransmitter(
                    name=input_.record, storage_url=storage_path, iens=iens
                )
            else:
                raise ValueError(f"Unsupported transmitter type: {storage_type}")
            futures.append(transmitter.transmit_data(record.data))
            transmitters[iens][input_.record] = transmitter
    asyncio.get_event_loop().run_until_complete(asyncio.gather(*futures))
    if isinstance(step_config, ert3.config.Unix):
        for command in step_config.transportable_commands:
            if storage_type == "shared_disk":
                transmitter = ert.data.SharedDiskRecordTransmitter(
                    name=command.name,
                    storage_path=pathlib.Path(storage_path),
                )
            elif storage_type == "ert_storage":
                transmitter = ert.storage.StorageRecordTransmitter(
                    name=command.name, storage_url=storage_path
                )
            else:
                raise ValueError(f"Unsupported transmitter type: {storage_type}")
            with open(command.location, "rb") as f:
                asyncio.get_event_loop().run_until_complete(
                    transmitter.transmit_data(f.read())
                )
            for iens in range(0, ensemble_size):
                transmitters[iens][command.name] = transmitter
    return dict(transmitters)


def _prepare_output(
    storage_type: str,
    step_config: Step,
    storage_path: str,
    ensemble_size: int,
) -> Dict[int, Dict[str, RecordTransmitter]]:
    transmitters: Dict[int, Dict[str, ert.data.RecordTransmitter]] = defaultdict(dict)

    for output in step_config.output:
        for iens in range(0, ensemble_size):
            if storage_type == "shared_disk":
                transmitters[iens][
                    output.record
                ] = ert.data.SharedDiskRecordTransmitter(
                    name=output.record,
                    storage_path=pathlib.Path(storage_path),
                )
            elif storage_type == "ert_storage":
                transmitters[iens][
                    output.record
                ] = ert.storage.StorageRecordTransmitter(
                    name=output.record, storage_url=storage_path, iens=iens
                )
            else:
                raise ValueError(f"Unsupported transmitter type: {storage_type}")
    return dict(transmitters)


def _build_ee_config(
    storage_path: str,
    ensemble: EnsembleConfig,
    stages_config: StagesConfig,
    input_records: RecordCollectionMap,
    dispatch_uri: str,
) -> Dict[str, Any]:
    if ensemble.size != None:
        ensemble_size = ensemble.size
    else:
        ensemble_size = input_records.ensemble_size

    stage = stages_config.step_from_key(ensemble.forward_model.stage)
    assert stage is not None
    commands = (
        stage.transportable_commands if isinstance(stage, ert3.config.Unix) else []
    )
    output_locations = [out.location for out in stage.output]
    jobs = []

    def command_location(name: str) -> FilePath:
        return next(
            (cmd.location for cmd in commands if cmd.name == name), pathlib.Path(name)
        )

    if isinstance(stage, ert3.config.Function):
        jobs.append(
            {
                "name": stage.function.__name__,
                "executable": cloudpickle.dumps(stage.function),
                "output": output_locations[0],
            }
        )

    if isinstance(stage, ert3.config.Unix):
        for script in stage.script:
            name, *args = script.split()
            jobs.append(
                {
                    "name": name,
                    "executable": command_location(name),
                    "args": tuple(args),
                }
            )

    steps = [
        {
            "name": stage.name + "-only_step",
            "inputs": [
                {
                    "record": input_.record,
                    "location": input_.location,
                    "mime": input_.mime,
                    "is_executable": False,
                }
                for input_ in stage.input
            ]
            + [
                {
                    "record": cmd.name,
                    "location": command_location(cmd.name),
                    "mime": cmd.mime,
                    "is_executable": True,
                }
                for cmd in commands
            ],
            "outputs": [
                {
                    "record": output.record,
                    "location": output.location,
                    "mime": output.mime,
                }
                for output in stage.output
            ],
            "jobs": jobs,
            "type": "function" if isinstance(stage, ert3.config.Function) else "unix",
        }
    ]

    ee_config: Dict[str, Any] = {
        "steps": steps,
        "realizations": ensemble_size,
        "max_running": 10000,
        "max_retries": 0,
        "executor": ensemble.forward_model.driver,
        "dispatch_uri": dispatch_uri,
    }

    assert ensemble_size is not None
    ee_config["inputs"] = _prepare_input(
        ensemble.storage_type, stage, input_records, storage_path, ensemble_size
    )
    ee_config["outputs"] = _prepare_output(
        ensemble.storage_type, stage, storage_path, ensemble_size
    )

    return ee_config


def _run(
    ensemble_evaluator: EnsembleEvaluator,
) -> Dict[int, Dict[str, RecordTransmitter]]:
    result = {}
    with ensemble_evaluator.run() as monitor:
        for event in monitor.track():
            if isinstance(event.data, dict) and event.data.get("status") in [
                _EVTYPE_SNAPSHOT_STOPPED,
                _EVTYPE_SNAPSHOT_FAILED,
            ]:
                monitor.signal_done()
            if event["type"] == EVTYPE_EE_TERMINATED and isinstance(event.data, bytes):
                result = pickle.loads(event.data)

    return result


def _prepare_output_records(
    raw_records: Dict[int, Dict[str, RecordTransmitter]]
) -> RecordCollectionMap:
    async def _load(
        iens: int, record_key: str, transmitter: RecordTransmitter
    ) -> Tuple[int, str, Record]:
        record = await transmitter.load()
        return (iens, record_key, record)

    futures = []
    for iens in sorted(raw_records.keys(), key=int):
        for record, transmitter in raw_records[iens].items():
            futures.append(_load(iens, record, transmitter))
    results = asyncio.get_event_loop().run_until_complete(asyncio.gather(*futures))

    data_results: Dict[int, Dict[str, Record]] = defaultdict(dict)
    for res in results:
        data_results[res[0]][res[1]] = res[2]

    output_records: Dict[str, List[Record]] = {
        rec_name: [] for rec_name in data_results[0]
    }
    for realization in data_results.values():
        assert output_records.keys() == realization.keys()
        for key in realization:
            output_records[key].append(realization[key])

    ensemble_records: Dict[str, RecordCollection] = {}
    for key in output_records:
        ensemble_records[key] = RecordCollection(records=output_records[key])

    return RecordCollectionMap(ensemble_records=ensemble_records)


def evaluate(
    workspace_root: Path,
    experiment_name: str,
    input_records: RecordCollectionMap,
    ensemble_config: EnsembleConfig,
    stages_config: StagesConfig,
) -> RecordCollectionMap:

    if ensemble_config.storage_type == "ert_storage":
        storage_path = ert.storage.get_records_url(workspace_root)
    else:
        evaluation_tmp_dir = _create_evaluator_tmp_dir(workspace_root, experiment_name)
        storage_path = str(evaluation_tmp_dir / ".my_storage")

    config = EvaluatorServerConfig()
    ee_config = _build_ee_config(
        storage_path,
        ensemble_config,
        stages_config,
        input_records,
        config.dispatch_uri,
    )
    ensemble = PrefectEnsemble(ee_config)  # type: ignore

    ee = EnsembleEvaluator(ensemble=ensemble, config=config, iter_=0)
    result = _run(ee)
    output_records = _prepare_output_records(result)

    return output_records


def cleanup(workspace_root: Path, evaluation_name: str) -> None:
    tmp_dir = _create_evaluator_tmp_dir(workspace_root, evaluation_name)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
