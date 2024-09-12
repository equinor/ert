from resdata.summary import Summary

import everest


def _compute_recovery_factor(load_case):
    summary = Summary(load_case)

    fopt = summary.last_value("FOPT")
    foip = summary.first_value("FOIP")

    if foip == 0:
        return 0

    return fopt / foip


def _save_object_value(object_value, target_file):
    with everest.jobs.io.safe_open(target_file, "w") as f:
        f.write("{}\n".format(object_value))


def recovery_factor(load_case, output_file):
    object_value = _compute_recovery_factor(load_case)
    _save_object_value(object_value, output_file)
