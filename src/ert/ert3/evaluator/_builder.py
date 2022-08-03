import pathlib
import shlex
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Type, cast

import cloudpickle
import ert.data
import ert.ensemble_evaluator
import ert.storage
from ert import ert3
from ert.async_utils import get_event_loop


def add_step_inputs(
    inputs: Tuple[ert3.config.LinkedInput, ...],
    transmitters: Dict[int, Dict[str, ert.data.RecordTransmitter]],
    step: ert.ensemble_evaluator.StepBuilder,
) -> None:
    for input_ in inputs:
        step_input = ert.ensemble_evaluator.InputBuilder().set_name(input_.name)

        if input_.stage_transformation:
            step_input.set_transformation(input_.stage_transformation)

        for iens, io_to_transmitter in transmitters.items():
            trans = io_to_transmitter[input_.name]
            # cast necessary due to https://github.com/python/mypy/issues/9656
            step_input.set_transmitter_factory(
                lambda _t=trans: cast(ert.data.RecordTransmitter, _t), iens
            )
        step.add_input(step_input)


def add_commands(
    stage: ert3.config.Unix,
    storage_type: str,
    storage_path: str,
    step: ert.ensemble_evaluator.StepBuilder,
) -> None:
    async def transform_output(
        transmitter: ert.data.RecordTransmitter, mime: str, location: pathlib.Path
    ) -> None:
        transformation = ert.data.ExecutableTransformation(location=location, mime=mime)
        record = await transformation.to_record()
        await transmitter.transmit_record(record)

    for command in stage.transportable_commands:
        transmitter: ert.data.RecordTransmitter
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
        get_event_loop().run_until_complete(
            transform_output(
                transmitter=transmitter,
                mime="application/octet-stream",
                location=command.location,
            )
        )
        step.add_input(
            ert.ensemble_evaluator.InputBuilder()
            .set_name(command.name)
            .set_transformation(
                ert.data.ExecutableTransformation(
                    location=stage.command_final_path_component(command.name)
                )
            )
            # cast necessary due to https://github.com/python/mypy/issues/9656
            .set_transmitter_factory(
                lambda _t=transmitter: cast(ert.data.RecordTransmitter, _t)
            )
        )


def add_step_outputs(
    storage_type: str,
    step_config: ert3.config.Step,
    storage_path: str,
    ensemble_size: int,
    step: ert.ensemble_evaluator.StepBuilder,
) -> None:
    for record_name, output in step_config.output.items():
        transformation = output.get_transformation_instance()
        output = ert.ensemble_evaluator.OutputBuilder().set_name(record_name)

        if transformation:
            output.set_transformation(transformation)

        for iens in range(0, ensemble_size):
            factory: Callable[
                [Type[ert.data.RecordTransmitter]], ert.data.RecordTransmitter
            ]
            if storage_type == "shared_disk":
                factory = partial(
                    ert.data.SharedDiskRecordTransmitter,
                    name=record_name,
                    storage_path=pathlib.Path(storage_path),
                )
            elif storage_type == "ert_storage":
                factory = partial(
                    ert.storage.StorageRecordTransmitter,
                    name=record_name,
                    storage_url=storage_path,
                    iens=iens,
                )
            else:
                raise ValueError(
                    f"unexpected storage type{storage_type} for {record_name} record"
                )
            output.set_transmitter_factory(factory, iens)
        step.add_output(output)


def build_ensemble(
    stage: ert3.config.Step,
    driver: str,
    ensemble_size: int,
    step_builder: ert.ensemble_evaluator.StepBuilder,
    active_mask: Optional[List[bool]] = None,
) -> ert.ensemble_evaluator.Ensemble:
    if active_mask is None:
        active_mask = [True] * ensemble_size
    if isinstance(stage, ert3.config.Function):
        step_builder.add_job(
            ert.ensemble_evaluator.JobBuilder()
            .set_name(stage.function.__name__)
            .set_index("0")
            .set_executable(cloudpickle.dumps(stage.function))
        )
    if isinstance(stage, ert3.config.Unix):
        for index, script in enumerate(stage.script):
            name, *args = shlex.split(script)
            step_builder.add_job(
                ert.ensemble_evaluator.JobBuilder()
                .set_executable(stage.command_final_path_component(name))
                .set_args(tuple(args))
                .set_name(name)
                .set_index(str(index))
            )

    builder = (
        ert.ensemble_evaluator.EnsembleBuilder()
        .set_ensemble_size(ensemble_size)
        .set_max_running(10000)
        .set_max_retries(0)
        .set_executor(driver)
    )
    for iens, active_flag in enumerate(active_mask):
        builder = builder.add_realization(
            ert.ensemble_evaluator.RealizationBuilder()
            .set_iens(iens)
            .active(active_flag)
            .add_step(step_builder)
        )

    return builder.build()
