import itertools
from typing import cast

import everest
from everest.config import EverestConfig
from everest.config.forward_model_config import SummaryResults


def extract_summary_keys(ever_config: EverestConfig) -> list[str]:
    summary_fms = [
        fm
        for fm in ever_config.forward_model
        if fm.results is not None and fm.results.type == "summary"
    ]

    if not summary_fms:
        return []

    summary_fm = summary_fms[0]
    assert summary_fm.results is not None

    smry_results = cast(SummaryResults, summary_fm.results)

    requested_keys: list[str] = ["*"] if smry_results.keys == "*" else smry_results.keys

    data_keys = everest.simulator.DEFAULT_DATA_SUMMARY_KEYS
    field_keys = everest.simulator.DEFAULT_FIELD_SUMMARY_KEYS
    well_sum_keys = everest.simulator.DEFAULT_WELL_SUMMARY_KEYS
    deprecated_user_specified_keys = (
        [] if ever_config.export is None else ever_config.export.keywords
    )

    wells = (
        [
            variable.name
            for control in ever_config.controls
            for variable in control.variables
            if control.type == "well_control"
        ]
        if ever_config.wells is None
        else [w.name for w in ever_config.wells]
    )

    well_keys = [
        f"{sum_key}:{wname}"
        for (sum_key, wname) in itertools.product(well_sum_keys, wells)
    ]

    all_keys = data_keys + field_keys + well_keys + deprecated_user_specified_keys

    return list(set(all_keys + requested_keys))
