from typing import Any

from resdata.summary import Summary

import everest


def _compute_recovery_factor(load_case: str) -> float:
    summary = Summary(load_case)

    fopt = summary.last_value("FOPT")
    foip = summary.first_value("FOIP")

    if foip == 0:
        return 0

    return fopt / foip


def _save_object_value(object_value: float, target_file: str) -> None:
    with everest.jobs.io.safe_open(target_file, "w") as f:
        f.write(f"{object_value}\n")


def recovery_factor(load_case: Any, output_file: str) -> None:
    object_value = _compute_recovery_factor(load_case)
    _save_object_value(object_value, output_file)
