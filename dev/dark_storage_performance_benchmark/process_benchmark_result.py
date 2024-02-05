#!/bin/env python3

from pathlib import Path
import json
from typing import List, Dict

OUTPUT_FILE_NAME = "processed_results.json"

with open(Path(__file__).with_name("output.json"), mode="rt", encoding="utf-8") as f:
    pytest_benchmark_output = json.load(f)

commit_id = pytest_benchmark_output["commit_info"]["id"]
benchmark_datetime = pytest_benchmark_output["datetime"]

results_dict = {
    "commit_id": commit_id,
    "datatime": benchmark_datetime,
    "benchmarks": {},
}
for benchmark in pytest_benchmark_output["benchmarks"]:
    results_dict["benchmarks"][benchmark["fullname"]] = benchmark["stats"]["mean"]

with open(Path(__file__).with_name(OUTPUT_FILE_NAME), mode="rt", encoding="utf-8") as f:
    results_json: List[Dict] = json.load(f)

for result in results_json:
    if result["commit_id"] == commit_id:
        raise SystemExit("Benchmark already registered")
results_json.append(results_dict)


with open(Path(__file__).with_name(OUTPUT_FILE_NAME), mode="wt", encoding="utf-8") as f:
    json.dump(results_json, f)
