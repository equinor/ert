"""
The experiment_server package provides facilities for creating, managing and
running experiments, as well as defining protocols for orderly and meaningful
communication between distributed parts of ERT.

This package defines the following:

 - An Experiment protocol [1] which all experiments are expected to implement
 - The communication protocols for communication between: 1) the client (GUI,
   CLI, and other stakeholders) and the server, 2) the server and remote
   workers.
 - an API for creating, managing and running experiments.

.. note::
   The experiment server is currently under active design and development,
   and is considered experimental.


[1] see https://peps.python.org/pep-0544/ for information about protocols
"""
from ._server import ExperimentServer

__all__ = ("ExperimentServer",)
