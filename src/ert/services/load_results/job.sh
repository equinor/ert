#!/usr/bin/env bash

# If the queue is local, load results will be performed centrally by ERT, rather
# than as part of the forward-model.
if [[ "${_ERT_QUEUE}" = "LOCAL" ]]; then
    echo "ERT is using LOCAL queue"
    echo "Loading of results will be done in the main process later"
    echo "Exiting with success"
    exit 0
fi

script_path="$(dirname $0)/__main__.py"
python_executable=$1
shift

exec "${python_executable}" "${script_path}" "$@"
