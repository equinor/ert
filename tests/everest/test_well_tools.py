import json
import os
import subprocess

import pytest
from ruamel.yaml import YAML

import everest


def _dump_sort_data(well_data_file, well_order_file):
    name = everest.ConfigKeys.NAME
    drill_time = everest.ConfigKeys.DRILL_TIME

    well_data = [
        {name: "PROD1", drill_time: 14},
        {name: "PROD2", drill_time: 11},
        {name: "INJECT1", drill_time: 1},
        {name: "INJECT2", drill_time: 114},
    ]

    with open(well_data_file, "w", encoding="utf-8") as f:
        json.dump(well_data, f)

    well_order = [
        "INJECT1",
        "PROD2",
        "INJECT2",
        "PROD1",
    ]

    with open(well_order_file, "w", encoding="utf-8") as f:
        json.dump(well_order, f)

    return [
        {name: "INJECT1", drill_time: 1},
        {name: "PROD2", drill_time: 11},
        {name: "INJECT2", drill_time: 114},
        {name: "PROD1", drill_time: 14},
    ]


def test_well_reorder(change_to_tmpdir):
    well_data_file = "well_data.json"
    well_order_file = "well_order.yml"
    output_file = "ordered_well_data.json"

    ordered_well_data = _dump_sort_data(
        well_data_file,
        well_order_file,
    )

    everest.jobs.well_tools.well_reorder(
        well_data_file,
        well_order_file,
        output_file,
    )

    with open(output_file, encoding="utf-8") as f:
        assert ordered_well_data == json.load(f)


@pytest.mark.integration_test
def test_well_reorder_script(change_to_tmpdir):
    assert os.access(everest.jobs.wdreorder, os.X_OK)

    well_data_file = "well_data.json"
    well_order_file = "well_order.yml"
    output_file = "ordered_well_data.json"

    ordered_well_data = _dump_sort_data(
        well_data_file,
        well_order_file,
    )

    cmd_fmt = "{well_reorder} --well_data {well_data} --order {order} --output {output}"
    cmd = cmd_fmt.format(
        well_reorder=everest.jobs.wdreorder,
        well_data=well_data_file,
        order=well_order_file,
        output=output_file,
    )

    subprocess.check_call(cmd, shell=True)

    with open(output_file, encoding="utf-8") as f:
        assert ordered_well_data == json.load(f)


def _dump_filter_data(well_data_file, well_filter_file):
    name = everest.ConfigKeys.NAME
    drill_time = everest.ConfigKeys.DRILL_TIME

    well_data = [
        {name: "PROD1", drill_time: 14},
        {name: "PROD2", drill_time: 11},
        {name: "INJECT1", drill_time: 1},
        {name: "INJECT2", drill_time: 114},
    ]

    with open(well_data_file, "w", encoding="utf-8") as f:
        json.dump(well_data, f)

    well_filter = [
        "INJECT1",
        "PROD2",
        "PROD1",
    ]

    with open(well_filter_file, "w", encoding="utf-8") as f:
        json.dump(well_filter, f)

    return [
        {name: "PROD1", drill_time: 14},
        {name: "PROD2", drill_time: 11},
        {name: "INJECT1", drill_time: 1},
    ]


def test_well_filter(change_to_tmpdir):
    well_data_file = "well_data.yml"
    well_filter_file = "well_filter.json"
    output_file = "ordered_well_data.json"

    destilled_well_data = _dump_filter_data(
        well_data_file,
        well_filter_file,
    )

    everest.jobs.well_tools.well_filter(
        well_data_file,
        well_filter_file,
        output_file,
    )

    with open(output_file, encoding="utf-8") as f:
        assert destilled_well_data == json.load(f)


@pytest.mark.integration_test
def test_well_filter_script(change_to_tmpdir):
    assert os.access(everest.jobs.wdfilter, os.X_OK)

    well_data_file = "well_data.json"
    well_filter_file = "well_filter.yml"
    output_file = "ordered_well_data.json"

    destilled_well_data = _dump_filter_data(
        well_data_file,
        well_filter_file,
    )

    cmd_fmt = (
        "{well_filter} --well_data {well_data} --filter {wfilter} --output {output}"
    )
    cmd = cmd_fmt.format(
        well_filter=everest.jobs.wdfilter,
        well_data=well_data_file,
        wfilter=well_filter_file,
        output=output_file,
    )

    subprocess.check_call(cmd, shell=True)

    with open(output_file, encoding="utf-8") as f:
        assert destilled_well_data == json.load(f)


def _dump_merge_data(well_data_file, additional_data_files):
    name = everest.ConfigKeys.NAME
    drill_time = everest.ConfigKeys.DRILL_TIME

    well_data = [
        {name: "PROD1", drill_time: 14},
        {name: "PROD2", drill_time: 11},
        {name: "INJECT1", drill_time: 1},
        {name: "INJECT2", drill_time: 114},
    ]

    with open(well_data_file, "w", encoding="utf-8") as f:
        json.dump(well_data, f)

    merged_data = [
        {name: "PROD1", drill_time: 14},
        {name: "PROD2", drill_time: 11},
        {name: "INJECT1", drill_time: 1},
        {name: "INJECT2", drill_time: 114},
    ]

    for idx, add_data_file in enumerate(additional_data_files):
        prop = "property_{}".format(idx)
        add_data = [
            {name: "PROD2", prop: idx * 100 + 11},
            {name: "INJECT1", prop: idx * 100 + 1},
            {name: "PROD1", prop: idx * 100 + 14},
            {name: "INJECT2", prop: idx * 100 + 114},
        ]

        yaml = YAML(typ="safe", pure=True)
        with open(add_data_file, "w", encoding="utf-8") as f:
            yaml.dump(add_data, f)

        merged_data[1][prop] = idx * 100 + 11
        merged_data[2][prop] = idx * 100 + 1
        merged_data[0][prop] = idx * 100 + 14
        merged_data[3][prop] = idx * 100 + 114

    return merged_data


@pytest.mark.integration_test
def test_well_update(change_to_tmpdir):
    assert os.access(everest.jobs.wdupdate, os.X_OK)

    well_data_file = "well_data.json"
    additional_data_files = (
        "well_prop_0.yml",
        "well_prop_1.yml",
        "well_prop_2.yml",
    )
    output_file = "ordered_well_data.json"

    for idx, _ in enumerate(additional_data_files):
        add_data_files = additional_data_files[: idx + 1]
        merge_well_data = _dump_merge_data(
            well_data_file,
            add_data_files,
        )

        cmd_fmt = (
            "{well_update} --well_data {well_data} "
            "--add_data {add_data} --output {output}"
        )
        cmd = cmd_fmt.format(
            well_update=everest.jobs.wdupdate,
            well_data=well_data_file,
            add_data=" ".join(add_data_files),
            output=output_file,
        )

        subprocess.check_call(cmd, shell=True)

        with open(output_file, encoding="utf-8") as f:
            load_data = json.load(f)
            assert merge_well_data == load_data


def test_well_set_invalid_data_length(change_to_tmpdir):
    name = everest.ConfigKeys.NAME

    well_data_file = "well_data.yml"
    well_data = [
        {name: "W1", "a": 1, "b": 2},
        {name: "W3", "a": 2, "b": 1},
        {name: "W2", "a": 7, "b": 5},
    ]
    yaml = YAML(typ="safe", pure=True)
    with open(well_data_file, "w", encoding="utf-8") as f:
        yaml.dump(well_data, f)

    new_data_file = "new_data.json"
    with open(new_data_file, "w", encoding="utf-8") as f:
        json.dump(new_data_file, f)

    output_file = "extended_well_data.json"
    with pytest.raises(ValueError):
        everest.jobs.well_tools.well_set(
            well_data_file,
            new_data_file,
            output_file,
        )


def test_well_set_too_many_entries(change_to_tmpdir):
    name = everest.ConfigKeys.NAME

    well_data_file = "well_data.yml"
    well_data = [
        {name: "W1", "a": 1, "b": 2},
        {name: "W3", "a": 2, "b": 1},
        {name: "W2", "a": 7, "b": 5},
    ]
    yaml = YAML(typ="safe", pure=True)
    with open(well_data_file, "w", encoding="utf-8") as f:
        yaml.dump(well_data, f)

    new_data_file = "new_data.json"
    with open(new_data_file, "w", encoding="utf-8") as f:
        json.dump(new_data_file, f)

    output_file = "extended_well_data.json"
    with pytest.raises(ValueError):
        everest.jobs.well_tools.well_set(
            well_data_file,
            new_data_file,
            output_file,
        )


def test_well_set_new_entry(change_to_tmpdir):
    name = everest.ConfigKeys.NAME

    well_data_file = "well_data.yml"
    well_data = [
        {name: "W1", "a": 1, "b": 2},
        {name: "W3", "a": 2, "b": 1},
        {name: "W2", "a": 7, "b": 5},
    ]
    yaml = YAML(typ="safe", pure=True)
    with open(well_data_file, "w", encoding="utf-8") as f:
        yaml.dump(well_data, f)

    new_data_file = "new_data.json"
    new_data = {"c": [4, 8, 12]}
    with open(new_data_file, "w", encoding="utf-8") as f:
        json.dump(new_data, f)

    output_file = "extended_well_data.json"
    everest.jobs.well_tools.well_set(
        well_data_file,
        new_data_file,
        output_file,
    )

    set_well_data = [
        {name: "W1", "a": 1, "b": 2, "c": 4},
        {name: "W3", "a": 2, "b": 1, "c": 8},
        {name: "W2", "a": 7, "b": 5, "c": 12},
    ]

    with open(output_file, encoding="utf-8") as f:
        assert set_well_data == json.load(f)


def test_well_set_entry(change_to_tmpdir):
    name = everest.ConfigKeys.NAME

    well_data_file = "well_data.yml"
    well_data = [
        {name: "W1", "a": 1, "b": 2},
        {name: "W3", "a": 2, "b": 1, "c": 100},
        {name: "W2", "a": 7, "b": 5},
    ]
    yaml = YAML(typ="safe", pure=True)
    with open(well_data_file, "w", encoding="utf-8") as f:
        yaml.dump(well_data, f)

    new_data_file = "new_data.json"
    new_data = {"c": [4, 8, 12]}
    with open(new_data_file, "w", encoding="utf-8") as f:
        json.dump(new_data, f)

    output_file = "extended_well_data.json"
    everest.jobs.well_tools.well_set(
        well_data_file,
        new_data_file,
        output_file,
    )

    set_well_data = [
        {name: "W1", "a": 1, "b": 2, "c": 4},
        {name: "W3", "a": 2, "b": 1, "c": 8},
        {name: "W2", "a": 7, "b": 5, "c": 12},
    ]

    with open(output_file, encoding="utf-8") as f:
        assert set_well_data == json.load(f)


@pytest.mark.integration_test
def test_well_set_script(change_to_tmpdir):
    assert os.access(everest.jobs.wdset, os.X_OK)
    name = everest.ConfigKeys.NAME

    well_data_file = "well_data.yml"
    well_data = [
        {name: "W1", "a": 1, "b": 2},
        {name: "W3", "a": 2, "b": 1},
        {name: "W2", "a": 7, "b": 5},
    ]
    yaml = YAML(typ="safe", pure=True)
    with open(well_data_file, "w", encoding="utf-8") as f:
        yaml.dump(well_data, f)

    new_data_file = "new_data.json"
    new_data = {"c": [4, 8, 12]}
    with open(new_data_file, "w", encoding="utf-8") as f:
        json.dump(new_data, f)

    output_file = "extended_well_data.json"

    cmd_fmt = "{well_set} --well_data {well_data} --entry {entry} --output {output}"
    cmd = cmd_fmt.format(
        well_set=everest.jobs.wdset,
        well_data=well_data_file,
        entry=new_data_file,
        output=output_file,
    )

    subprocess.check_call(cmd, shell=True)

    set_well_data = [
        {name: "W1", "a": 1, "b": 2, "c": 4},
        {name: "W3", "a": 2, "b": 1, "c": 8},
        {name: "W2", "a": 7, "b": 5, "c": 12},
    ]

    with open(output_file, encoding="utf-8") as f:
        assert set_well_data == json.load(f)


def _dump_completion_data(well_data_file, start_date):
    name = everest.ConfigKeys.NAME
    drill_time = everest.ConfigKeys.DRILL_TIME
    drill_date = everest.ConfigKeys.DRILL_DATE
    drill_delay = everest.ConfigKeys.DRILL_DELAY
    completion_date = everest.ConfigKeys.COMPLETION_DATE

    well_data = [
        {name: "PROD1", drill_time: 30, drill_date: "2001-01-01"},
        {name: "PROD2", drill_time: 40, drill_date: "2001-01-01"},
        {name: "PROD3", drill_time: 2, drill_date: "2001-03-01"},
        {name: "INJECT1", drill_date: "2001-03-01"},
        {name: "INJECT2", drill_time: 400, drill_date: "2002-01-01"},
        {name: "INJECT3", drill_time: 30, drill_date: "2003-01-01"},
        {name: "INJECT4", drill_time: 30},
        {name: "INJECT5", drill_time: 1, drill_delay: 3},
        {name: "INJECT6", drill_time: 1, drill_delay: 4, drill_date: "2003-04-15"},
    ]

    yaml = YAML(typ="safe", pure=True)
    with open(well_data_file, "w", encoding="utf-8") as f:
        yaml.dump(well_data, f)

    return [
        {
            name: "PROD1",
            drill_time: 30,
            drill_date: "2001-01-01",
            completion_date: "2001-02-01",
        },
        {
            name: "PROD2",
            drill_time: 40,
            drill_date: "2001-01-01",
            completion_date: "2001-03-13",
        },
        {
            name: "PROD3",
            drill_time: 2,
            drill_date: "2001-03-01",
            completion_date: "2001-03-15",
        },
        {
            name: "INJECT1",
            drill_date: "2001-03-01",
            completion_date: "2001-03-15",
        },
        {
            name: "INJECT2",
            drill_time: 400,
            drill_date: "2002-01-01",
            completion_date: "2003-02-05",
        },
        {
            name: "INJECT3",
            drill_time: 30,
            drill_date: "2003-01-01",
            completion_date: "2003-03-07",
        },
        {
            name: "INJECT4",
            drill_time: 30,
            completion_date: "2003-04-06",
        },
        {
            name: "INJECT5",
            drill_time: 1,
            drill_delay: 3,
            completion_date: "2003-04-10",
        },
        {
            name: "INJECT6",
            drill_time: 1,
            drill_delay: 4,
            drill_date: "2003-04-15",
            completion_date: "2003-04-16",
        },
    ]


def test_add_completion_date(change_to_tmpdir):
    well_data_file = "well_data.yml"
    output_file = "humble_wells.json"
    start_date = "2001-01-02"

    completion_data = _dump_completion_data(
        well_data_file,
        start_date,
    )

    everest.jobs.well_tools.add_completion_date(
        well_data_file,
        start_date,
        output_file,
    )

    with open(output_file, encoding="utf-8") as f:
        assert completion_data == json.load(f)


@pytest.mark.integration_test
def test_completion_date_script(change_to_tmpdir):
    assert os.access(everest.jobs.wdcompl, os.X_OK)

    well_data_file = "well_data.yml"
    output_file = "humble_wells.json"

    start_date = "2001-01-02"
    completion_data = _dump_completion_data(
        well_data_file,
        start_date,
    )

    cmd_fmt = (
        "{add_completion_date} --well_data {well_data} "
        "--start_date {start_date} --output {output}"
    )
    cmd = cmd_fmt.format(
        add_completion_date=everest.jobs.wdcompl,
        well_data=well_data_file,
        start_date=start_date,
        output=output_file,
    )

    subprocess.check_call(cmd, shell=True)

    with open(output_file, encoding="utf-8") as f:
        assert completion_data == json.load(f)


def _dump_compl_filter_data(well_data_file):
    name = everest.ConfigKeys.NAME
    drill_time = everest.ConfigKeys.DRILL_TIME
    drill_date = everest.ConfigKeys.DRILL_DATE
    completion_date = everest.ConfigKeys.COMPLETION_DATE

    well_data = [
        {
            name: "PROD1",
            drill_time: 30,
            drill_date: "2001-01-01",
            completion_date: "2001-02-01",
        },
        {
            name: "PROD2",
            drill_time: 40,
            drill_date: "2001-01-01",
            completion_date: "2001-03-13",
        },
        {
            name: "PROD3",
            drill_time: 2,
            drill_date: "2001-03-01",
            completion_date: "2001-03-15",
        },
        {
            name: "INJECT1",
            drill_date: "2001-03-01",
            completion_date: "2001-03-15",
        },
        {
            name: "INJECT2",
            drill_time: 400,
            drill_date: "2002-01-01",
            completion_date: "2003-02-05",
        },
        {
            name: "INJECT3",
            drill_time: 30,
            drill_date: "2003-01-01",
            completion_date: "2003-03-07",
        },
        {
            name: "INJECT4",
            drill_time: 30,
            completion_date: "2003-04-06",
        },
    ]

    yaml = YAML(typ="safe", pure=True)
    with open(well_data_file, "w", encoding="utf-8") as f:
        yaml.dump(well_data, f)

    filtered_well_data = [
        {
            name: "PROD2",
            drill_time: 40,
            drill_date: "2001-01-01",
            completion_date: "2001-03-13",
        },
        {
            name: "PROD3",
            drill_time: 2,
            drill_date: "2001-03-01",
            completion_date: "2001-03-15",
        },
        {
            name: "INJECT1",
            drill_date: "2001-03-01",
            completion_date: "2001-03-15",
        },
        {
            name: "INJECT2",
            drill_time: 400,
            drill_date: "2002-01-01",
            completion_date: "2003-02-05",
        },
        {
            name: "INJECT3",
            drill_time: 30,
            drill_date: "2003-01-01",
            completion_date: "2003-03-07",
        },
    ]

    return filtered_well_data, "2001-01-30", "2003-03-07"


def test_filter_completion_date(change_to_tmpdir):
    well_data_file = "well_data.yml"
    output_file = "filtered_well_data.json"

    case_data = _dump_compl_filter_data(well_data_file)
    filtered_data, start_date, end_date = case_data

    everest.jobs.well_tools.well_opdate_filter(
        well_data_file,
        start_date,
        end_date,
        output_file,
    )

    with open(output_file, encoding="utf-8") as f:
        assert filtered_data == json.load(f)


@pytest.mark.integration_test
def test_filter_completion_date_script(change_to_tmpdir):
    assert os.access(everest.jobs.wddatefilter, os.X_OK)

    well_data_file = "well_data.yml"
    output_file = "filtered_well_data.json"

    case_data = _dump_compl_filter_data(well_data_file)
    filtered_data, start_date, end_date = case_data

    cmd_fmt = (
        "{datefilter} --well_data {well_data} --start_date {start_date} "
        "--end_date={end_date} --output {output}"
    )
    cmd = cmd_fmt.format(
        datefilter=everest.jobs.wddatefilter,
        well_data=well_data_file,
        start_date=start_date,
        end_date=end_date,
        output=output_file,
    )

    subprocess.check_call(cmd, shell=True)

    with open(output_file, encoding="utf-8") as f:
        assert filtered_data == json.load(f)
