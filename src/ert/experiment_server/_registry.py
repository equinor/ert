from typing import Dict, List

from ._experiment_protocol import Experiment


class _Registry:
    """:class:`_Registry` is a registry for registering experiments to be run
    or managed by the :class:`ert.experiment_server._server.ExperimentServer`.
    The registry should be manageable, but also be queryable by the
    :class:`ert.experiment_server._server.ExperimentServer`. The class's
    purpose is to decouple grunt tasks like managing the registry from the more
    complex task of implementing an API on top of it.
    """

    def __init__(self) -> None:
        self._experiments: Dict[str, Experiment] = {}

    def add_experiment(self, experiment: Experiment) -> None:
        """Add an experiment to the registry.

        An id is generated here, but it should share (or receive) this ID from
        ert-storage. See [1].

        [1] https://github.com/equinor/ert/issues/3437#issue-1247962008
        """
        self._experiments[experiment.id_] = experiment

    @property
    def all_experiments(self) -> List[Experiment]:
        return list(self._experiments.values())

    def get_experiment(self, experiment_id: str) -> Experiment:
        return self._experiments[experiment_id]
