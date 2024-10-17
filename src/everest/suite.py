from __future__ import annotations

import json
import logging
import queue
import random
from typing import Any, Dict

from ert.config import QueueSystem
from ert.ensemble_evaluator import EvaluatorServerConfig
from ert.run_models import StatusEvents
from ert.run_models.batch_simulator_run_model import BatchSimulatorRunModel
from ert.storage import open_storage
from everest.config import EverestConfig
from everest.plugins.site_config_env import PluginSiteConfigEnv
from everest.simulator.everest_to_ert import everest_to_ert_config
from everest.strings import EVEREST
from everest.util import makedirs_if_needed


def start_optimization(
    config, simulation_callback=None, optimization_callback=None, display_all_jobs=True
):
    workflow = _EverestWorkflow(
        config, simulation_callback, optimization_callback, display_all_jobs
    )
    with PluginSiteConfigEnv():
        res = workflow.start_optimization()
    return res


def _add_defaults(config: EverestConfig):
    """This function exists as a temporary mechanism to default configurations that
    needs to be global in the sense that they should carry over both to ropt and ERT.
    When the proper mechanism for this is implemented this code
    should die.

    """
    defaulted_config = config.copy()
    assert defaulted_config.environment is not None

    random_seed = defaulted_config.environment.random_seed
    if random_seed is None:
        random_seed = random.randint(1, 2**30)

    defaulted_config.environment.random_seed = random_seed

    logging.getLogger(EVEREST).info("Using random seed: %d", random_seed)
    logging.getLogger(EVEREST).info(
        "To deterministically reproduce this experiment, "
        "add the above random seed to your configuration file."
    )

    return defaulted_config


class _EverestWorkflow(object):
    """
    An instance of this class is the main object in everest.

    Through this object an optimization experiment is instantiated and executed/run.
    This object will provide access to the entire optimization configuration.
    """

    def __init__(
        self,
        config: EverestConfig,
        simulation_callback=None,
        optimization_callback=None,
        display_all_jobs=True,
    ):
        """Will initialize an Everest instance either from a configuration file or
        a loaded config.

        @config   a dictionary containing the configuration.  See everest --doc
                  for documentation on the config

        @callback a function that will be called whenever changes in the
                  simulation or optimization routine occur, e.g., when one
                  realization's simulation completes, the status vector will be
                  sent, with the event SIMULATOR_UPDATE.
        """

        # Callbacks
        self._sim_callback = simulation_callback
        self._opt_callback = optimization_callback

        self._monitor_thread = None  # Thread for monitoring simulator activity

        self._config = _add_defaults(config)

        makedirs_if_needed(self.config.log_dir)
        makedirs_if_needed(self.config.optimization_output_dir)

        self._simulation_delete_run_path = (
            False
            if config.simulator is None
            else (config.simulator.delete_run_path or False)
        )

        self._display_all_jobs = display_all_jobs
        self._fm_errors: Dict[str, Dict[str, Any]] = {}
        self._max_batch_num_reached = False

    def start_optimization(self):
        """Run an optimization with the current settings.

        This method must be called from the same thread where this
        object has been created (probably because of the use of sqlite3
        deeper down).
        This method is not thread safe. Multiple overlapping executions
        of this method will probably lead to a crash
        """
        assert self._monitor_thread is None

        ert_config = everest_to_ert_config(self.config)
        with open_storage(ert_config.ens_path, mode="w") as storage:
            status_queue: queue.SimpleQueue[StatusEvents] = queue.SimpleQueue()

            run_model = BatchSimulatorRunModel(
                random_seed=ert_config.random_seed,
                config=ert_config,
                everest_config=self.config,
                storage=storage,
                status_queue=status_queue,
                simulation_callback=self._sim_callback,
                optimization_callback=self._opt_callback,
            )

            evaluator_server_config = EvaluatorServerConfig(
                custom_port_range=range(49152, 51819)
                if ert_config.queue_config.queue_system == QueueSystem.LOCAL
                else None
            )

            to_return_value = run_model.run_experiment(evaluator_server_config)

            # Extract the best result from the storage.
            self._result = run_model._result

            return to_return_value

    @property
    def result(self):
        return self._result

    @property
    def config(self) -> EverestConfig:
        return self._config

    def __repr__(self):
        return "EverestWorkflow(config=%s)" % json.dumps(
            self.config, sort_keys=True, indent=2
        )
