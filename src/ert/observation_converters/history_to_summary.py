import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

from ert.config.observation_config_migrations import (
    remove_refcase_and_time_map_dependence_from_obs_config,
)
from ert.namespace import Namespace
from ert.plugins import ErtRuntimePlugins

logger = logging.getLogger(__name__)


def run_convert_observations(
    args: Namespace, _: ErtRuntimePlugins | None = None
) -> None:
    changes = remove_refcase_and_time_map_dependence_from_obs_config(args.config)

    if changes is None or changes.is_empty():
        logger.info("convert_observations did not make any changes")
        print(
            "No observations dependent on TIME_MAP / REFCASE found, you can "
            "safely remove TIME_MAP / REFCASE and the "
            "corresponding files from ERT config."
        )
        return

    obs_config_to_edit_path = changes.obs_config_path + ".updated"
    print(
        f"Making copy of obs config "
        f"@ {changes.obs_config_path} -> {obs_config_to_edit_path}"
    )

    shutil.copy(changes.obs_config_path, obs_config_to_edit_path)
    print(f"Applying change to obs config @ {obs_config_to_edit_path}...")
    changes.apply_to_file(Path(obs_config_to_edit_path))
    convert_observations_trace = ""
    for history_change in changes.history_changes:
        convert_observations_trace += (
            f"History obs {history_change.source_observation.name} "
            f"-> {len(history_change.summary_obs_declarations)} summary observations\n"
        )

    for gen_obs_change in changes.general_obs_changes:
        convert_observations_trace += (
            f"General obs {gen_obs_change.source_observation.name}, changing "
            f"DATE {gen_obs_change.source_observation.date} "
            f"to RESTART={gen_obs_change.restart}\n"
        )
    for summary_change in changes.summary_obs_changes:
        convert_observations_trace += (
            f"Summary obs {summary_change.source_observation.name}, changing "
            f"RESTART {summary_change.source_observation.restart} "
            f"to DATE={summary_change.date}\n"
        )

    logger.info(f"convert_observations trace: \n {convert_observations_trace}")
    print(convert_observations_trace)

    os.rename(
        changes.obs_config_path,
        f"{changes.obs_config_path}-{datetime.now().astimezone().strftime('%Y-%m-%d-%H-%M-%S')}.old",
    )
    os.rename(obs_config_to_edit_path, changes.obs_config_path)
    msg = (
        f"Observation changes applied to {changes.obs_config_path}. The old "
        f"observations file is now at {changes.obs_config_path}.old and can be "
        f"safely deleted if the new one works."
    )
    print(msg)
    logger.info(msg)
