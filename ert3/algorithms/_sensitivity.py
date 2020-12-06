def _build_base_records(groups, parameters):
    return {group_name: parameters[group_name].ppf(0.5) for group_name in groups}


def one_at_the_time(parameters):
    tail = (1 - 0.99) / 2

    if len(parameters) == 0:
        raise ValueError("Cannot study the sensitivity of no variables")

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
                rec_values[idx] = lim_val[idx]
                records[group_name] = rec_values

                evaluations.append(records)

    return evaluations
