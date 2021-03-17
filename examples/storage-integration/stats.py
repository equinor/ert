import json


def analyze_file(filename, record_name, fields):
    with open(filename) as f:
        data = json.load(f)

    timing = {f: {"min": 100000, "max": 0, "sum": 0} for f in fields}

    for real in data:
        for w in fields:
            timing[w]["sum"] += real["output"][record_name][w + "_time"]
            timing[w]["min"] = min(
                timing[w]["min"], real["output"][record_name][w + "_time"]
            )
            timing[w]["max"] = max(
                timing[w]["max"], real["output"][record_name][w + "_time"]
            )

    for w in fields:
        timing[w]["average"] = timing[w]["sum"] / len(data)
        print(w)
        for k, v in timing[w].items():
            print(f"\t{k}: {v} seconds")


print("publish")
analyze_file(
    filename="experiments/publish/data.json",
    record_name="something_new",
    fields=["execution", "upload", "matrix_upload"],
)
print("fetch")
analyze_file(
    filename="experiments/fetch/data.json",
    record_name="something_else",
    fields=["execution", "published_file", "published_matrix"],
)
print("direct")
analyze_file(
    filename="experiments/publish_direct/data.json",
    record_name="direct_times",
    fields=["upload"],
)
