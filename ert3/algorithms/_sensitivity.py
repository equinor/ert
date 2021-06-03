from typing import Set, MutableMapping, List, Dict

from ert3.stats import Distribution
from ert3.data import Record


def _build_base_records(
    groups: Set[str], parameters: MutableMapping[str, Distribution]
) -> Dict[str, Record]:
    return {group_name: parameters[group_name].ppf(0.5) for group_name in groups}


def one_at_the_time(
    parameters: MutableMapping[str, Distribution]
) -> List[Dict[str, Record]]:
    if len(parameters) == 0:
        raise ValueError("Cannot study the sensitivity of no variables")

    # The lower extremal value of the analysis. Each variable will after turn
    # be explored as ppf(tail) and ppf(1-tail) as the two extremal values.
    tail = (1 - 0.99) / 2

    evaluations = []
    for group_name, dist in parameters.items():
        lower = dist.ppf(tail)
        upper = dist.ppf(1 - tail)
        const_records = set(parameters.keys())
        const_records.remove(group_name)

        for idx in dist.index:
            for lim_val in (lower, upper):
                records = _build_base_records(const_records, parameters)

                rec_values = dist.ppf(0.5)
                rec_values.data[idx] = lim_val.data[idx]  # type: ignore
                records[group_name] = rec_values

                evaluations.append(records)

    return evaluations
