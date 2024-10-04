from typing import Any, List, Optional, Tuple

from ert.config import QueueSystem
from everest.config import EverestConfig
from everest.config.simulator_config import SimulatorConfig
from everest.config_keys import ConfigKeys

_LSF_OPTIONS = [
    (ConfigKeys.CORES, "MAX_RUNNING"),
    (ConfigKeys.LSF_QUEUE_NAME, "LSF_QUEUE"),
    (ConfigKeys.LSF_SERVER, "LSF_SERVER"),
    (ConfigKeys.LSF_OPTIONS, "LSF_RESOURCE"),
]

_SLURM_OPTIONS = [
    (ConfigKeys.CORES, "MAX_RUNNING"),
    (ConfigKeys.SLURM_QUEUE, "PARTITION"),
    (ConfigKeys.SLURM_SBATCH, "SBATCH"),
    (ConfigKeys.SLURM_SCANCEL, "SCANCEL"),
    (ConfigKeys.SLURM_SCONTROL, "SCONTROL"),
    (ConfigKeys.SLURM_SQUEUE, "SQUEUE"),
    (ConfigKeys.SLURM_MAX_RUNTIME, "MAX_RUNTIME"),
    (ConfigKeys.SLURM_MEMORY, "MEMORY"),
    (ConfigKeys.SLURM_MEMORY_PER_CPU, "MEMORY_PER_CPU"),
    (ConfigKeys.SLURM_SQUEUE_TIMEOUT, "SQUEUE_TIMEOUT"),
    (ConfigKeys.SLURM_EXCLUDE_HOST_OPTION, "EXCLUDE_HOST"),
    (ConfigKeys.SLURM_INCLUDE_HOST_OPTION, "INCLUDE_HOST"),
]


def _extract_ert_queue_options_from_simulator_config(
    simulator: Optional[SimulatorConfig], queue_system
) -> List[Tuple[str, str, Any]]:
    if simulator is None:
        simulator = SimulatorConfig()

    if queue_system == ConfigKeys.LSF:
        return simulator.extract_ert_queue_options(
            queue_system=QueueSystem.LSF, everest_to_ert_key_tuples=_LSF_OPTIONS
        )
    elif queue_system == ConfigKeys.LOCAL:
        return [
            (
                QueueSystem.LOCAL,
                "MAX_RUNNING",
                simulator.cores or 8,
            )
        ]
    elif queue_system == ConfigKeys.SLURM:
        return simulator.extract_ert_queue_options(
            queue_system=QueueSystem.SLURM, everest_to_ert_key_tuples=_SLURM_OPTIONS
        )

    raise KeyError(
        f"Invalid queue_system: {queue_system}, "
        "expected one of: ['lsf', 'local', 'slurm']"
    )


def _extract_queue_system(ever_config: EverestConfig, ert_config):
    queue_system = (
        ever_config.simulator.queue_system if ever_config.simulator else None
    ) or "local"
    ert_config["QUEUE_SYSTEM"] = QueueSystem(queue_system.upper())
    ert_config.setdefault("QUEUE_OPTION", []).extend(
        _extract_ert_queue_options_from_simulator_config(
            ever_config.simulator, queue_system
        )
    )
