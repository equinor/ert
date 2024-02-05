#!/bin/env python3
import json
from typing import Dict, Any, Union, List
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import colormaps

def plot_marketplace_action_output():
    with open(Path(__file__).with_name("data.json"), "r+", encoding="utf-8") as f:
        data: Dict[str, str] = json.load(f)

    data_entries = data["entries"]
    dark_storage_benchmark_results: Dict[str, Any] = data_entries[
        "Python Dark Storage Benchmark"
    ]

    cmap = plt.cm.get_cmap("viridis")

    test_and_result_dict: Dict[str, Union[str, Dict]] = defaultdict(dict)

    for benchmark_result in dark_storage_benchmark_results:
        commit_id = benchmark_result["commit"]["id"]

        for test in benchmark_result["benches"]:
            test_name = test["name"]
            test_value = test["value"]
            test_dict = test_and_result_dict[test_name]

            # first pass
            if test_dict.get("test_modifier_constant") is None:
                test_dict["test_modifier_constant"] = 1 / test_value
                test_dict["results"] = []
                test_dict["total_value"] = 0
                test_dict["total_tests"] = 0
            test_dict["results"].append(
                {
                    "commit_id": commit_id,
                    "value": test_value * test_dict["test_modifier_constant"],
                }
            )
            test_dict["total_value"] += test_value * test_dict["test_modifier_constant"]
            test_dict["total_tests"] += 1

    COMMITS_TO_SHOW = 8
    commit_labels = []
    for test_index, test_data in enumerate(test_and_result_dict.values()):
        for order_index, test_result in enumerate(
            test_data["results"][-COMMITS_TO_SHOW:]
        ):
            commit_id = test_result["commit_id"][:7]
            if commit_id not in commit_labels:
                commit_labels.append(commit_id)
            color = cmap(order_index / COMMITS_TO_SHOW)
            plt.scatter(test_index, test_result["value"], color=color, label=commit_id)

        plt.scatter(
            test_index,
            test_data["total_value"] / test_data["total_tests"],
            color="red",
            label="mean",
            s=70,
            edgecolors="black",
        )

    plt.ylim(top=1.30, bottom=0.70)
    plt.grid(True)
    plt.xlabel("Test Index")
    plt.ylabel("Value relative to baseline")
    plt.legend(
        labels=commit_labels + ["mean"], loc="lower center", ncol=len(commit_labels) + 1
    )
    plt.show()


def plot_custom_action_output():
    FIGSIZE = [19, 10]
    VARIANCE = 0.25
    with open(
        Path(__file__).with_name("processed_results.json"), "r+", encoding="utf-8"
    ) as f:
        dark_storage_benchmark_results: List[Dict] = json.load(f)

    cmap = colormaps.get_cmap("viridis") #plt.cm.get_cmap("viridis")

    test_and_result_dict: Dict[str, Union[str, Dict]] = defaultdict(dict)
    plt.figure(figsize=FIGSIZE)
    for benchmark_result in dark_storage_benchmark_results:
        commit_id = benchmark_result["commit_id"]

        for test_name, test_value in benchmark_result["benchmarks"].items():
            test_dict = test_and_result_dict[test_name]

            # first pass
            if test_dict.get("test_modifier_constant") is None:
                test_dict["test_modifier_constant"] = 1 / test_value
                test_dict["results"] = []
                test_dict["total_value"] = 0
                test_dict["total_tests"] = 0
            test_dict["results"].append(
                {
                    "commit_id": commit_id,
                    "value": test_value * test_dict["test_modifier_constant"],
                }
            )
            test_dict["total_value"] += test_value * test_dict["test_modifier_constant"]
            test_dict["total_tests"] += 1

    COMMITS_TO_SHOW = 8
    commit_labels = []
    for test_index, test_data in enumerate(test_and_result_dict.values()):
        for order_index, test_result in enumerate(
            test_data["results"][-COMMITS_TO_SHOW:]
        ):
            commit_id = test_result["commit_id"][:7]
            if commit_id not in commit_labels:
                commit_labels.append(commit_id)
            color = cmap(order_index / COMMITS_TO_SHOW)
            plt.scatter(test_index, test_result["value"], color=color, label=commit_id)

        plt.scatter(
            test_index,
            test_data["total_value"] / test_data["total_tests"],
            color="red",
            label="mean",
            s=70,
            edgecolors="black",
        )

    plt.ylim(top=1.00 + VARIANCE, bottom=1.00 - VARIANCE)
    plt.grid(True)
    plt.xlabel("Test Index")
    plt.ylabel("Value relative to baseline")
    plt.legend(
        labels=commit_labels + ["mean"], loc="lower center", ncol=len(commit_labels) + 1
    )
    plt.savefig(Path(__file__).with_name("benchmark.svg"))


if __name__ == "__main__":
    plot_custom_action_output()
