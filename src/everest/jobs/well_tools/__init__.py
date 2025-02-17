import datetime
import json
from typing import Any

import everest


def well_reorder(well_data_file: str, well_order_file: str, output_file: str) -> None:
    well_data = everest.jobs.io.load_data(well_data_file)
    well_order = everest.jobs.io.load_data(well_order_file)
    well_data.sort(key=lambda well: well_order.index(well["name"]))

    with everest.jobs.io.safe_open(output_file, "w") as fout:
        json.dump(well_data, fout)


def well_filter(well_data_file: str, well_filter_file: str, output_file: str) -> None:
    well_data = everest.jobs.io.load_data(well_data_file)
    well_filter = everest.jobs.io.load_data(well_filter_file)
    well_data = [
        well_entry for well_entry in well_data if well_entry["name"] in well_filter
    ]

    with everest.jobs.io.safe_open(output_file, "w") as fout:
        json.dump(well_data, fout)


def well_update(
    master_data_file: str, additional_data_files: list[str], output_file: str
) -> None:
    well_data = everest.jobs.io.load_data(master_data_file)

    for add_data_file in additional_data_files:
        add_data = everest.jobs.io.load_data(add_data_file)
        add_data = {well_entry["name"]: well_entry for well_entry in add_data}

        for well_entry in well_data:
            well_entry.update(add_data[well_entry["name"]])

    with everest.jobs.io.safe_open(output_file, "w") as fout:
        json.dump(well_data, fout)


def well_set(well_data_file: str, new_entry_file: str, output_file: str) -> None:
    well_data = everest.jobs.io.load_data(well_data_file)
    new_entry = everest.jobs.io.load_data(new_entry_file)

    if len(new_entry) != 1:
        err_msg = "Expected there to be exactly one new entry in {nef}, was {ne}"
        raise ValueError(err_msg.format(nef=new_entry_file, ne=len(new_entry)))

    entry_key = next(iter(new_entry.keys()))
    entry_data = new_entry[entry_key]

    if len(well_data) != len(entry_data):
        err_msg = f"Expected number of entries in {well_data_file} to be equal "
        "the number of entries in {new_entry_file} ({nwell} != {nentry})"
        err_msg = err_msg.format(
            well_data_file=well_data_file,
            new_entry_file=new_entry_file,
            nwell=len(well_data),
            nwells=len(entry_data),
        )
        raise ValueError(err_msg)

    for well_entry, data_elem in zip(well_data, entry_data, strict=False):
        well_entry[entry_key] = data_elem

    with everest.jobs.io.safe_open(output_file, "w") as fout:
        json.dump(well_data, fout)


def add_completion_date(
    well_data_file: str, start_date_str: str, output_file: str
) -> None:
    start_date = everest.util.str2date(start_date_str)
    well_data = everest.jobs.io.load_data(well_data_file)

    prev_date = start_date
    for well_entry in well_data:
        well_start_date = well_entry.get("drill_date")
        if well_start_date is not None:
            well_start_date = everest.util.str2date(well_start_date)
        else:
            well_start_date = prev_date

        drill_delay = well_entry.get("drill_delay", 0)
        well_start_date = max(
            well_start_date,
            prev_date + datetime.timedelta(days=drill_delay),
        )

        drill_time = well_entry.get("drill_time", 0)
        completion_date = well_start_date + datetime.timedelta(days=drill_time)
        well_entry["completion_date"] = everest.util.date2str(completion_date)
        prev_date = completion_date

    with everest.jobs.io.safe_open(output_file, "w") as fout:
        json.dump(well_data, fout)


def _valid_operational_dates(
    well_entry: Any, start_date: datetime.datetime, end_date: datetime.datetime
) -> bool:
    drill_time = well_entry.get("drill_time", 0)
    compl_date = everest.util.str2date(well_entry["completion_date"])
    real_drill_date = compl_date - datetime.timedelta(days=drill_time)

    return start_date <= real_drill_date <= compl_date <= end_date


def well_opdate_filter(
    well_data_file: str, start_date_str: str, end_date_str: str, output_file: str
) -> None:
    start_date = everest.util.str2date(start_date_str)
    end_date = everest.util.str2date(end_date_str)
    well_data = everest.jobs.io.load_data(well_data_file)

    # pylint: disable=unnecessary-lambda-assignment
    valid = lambda well: _valid_operational_dates(well, start_date, end_date)
    well_data = list(filter(valid, well_data))

    with everest.jobs.io.safe_open(output_file, "w") as fout:
        json.dump(well_data, fout)
