from typing import Any, Dict, Tuple, Union
from ert_shared.ensemble_evaluator.config import find_open_port

PBS_CLUSTER_CLASS: str = "dask_jobqueue.LSFCluster"
PBS_CLUSTER_KWARGS: Dict[str, Any] = {
    "n_workers": 10,
    "queue": "normal",
    "project": "ERT-TEST",
    "local_directory": "$TMPDIR",
    "cores": 4,
    "memory": "16GB",
    "resource_spec": "select=1:ncpus=4:mem=16GB",
}

LSF_CLUSTER_CLASS: str = "dask_jobqueue.PBSCluster"
LSF_CLUSTER_KWARGS: Dict[str, Any] = {
    "queue": "mr",
    "project": None,
    "cores": 1,
    "memory": "1GB",
    "use_stdin": True,
    "n_workers": 2,
    "silence_logs": "debug",
    "scheduler_options": {"port": find_open_port()},  # not nice, but file not __init__
}

LOCAL_EXECUTOR: str = "local"
LSF_EXECUTOR: str = "lsf"
PBS_EXECUTOR: str = "pbs"
DASK_EXECUTORS: Tuple[str, str, str] = (
    LOCAL_EXECUTOR,
    LSF_EXECUTOR,
    PBS_EXECUTOR,
)

LOCAL_DASK_EXECUTOR_KWARGS: Dict[str, Union[str, Dict[str, int]]] = {
    "silence_logs": "debug",
    "scheduler_options": {"port": find_open_port()},
}

PBS_DASK_EXECUTOR_KWARGS: Dict[str, Any] = {
    "cluster_class": PBS_CLUSTER_CLASS,
    "cluster_kwargs": PBS_CLUSTER_KWARGS,
    "debug": True,
}

LSF_DASK_EXECUTOR_KWARGS: Dict[str, Any] = {
    "cluster_class": LSF_CLUSTER_CLASS,
    "cluster_kwargs": LSF_CLUSTER_KWARGS,
    "debug": True,
}

RETRY_DELAY: int = 5  # seconds
