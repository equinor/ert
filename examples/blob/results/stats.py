import json


def analyze_file(filename, fields):
    with open(filename) as f:
        data = json.load(f)

    timing = {f: {"min": 100000, "max": 0, "sum": 0} for f in fields}

    for real in data:
        for w in fields:
            timing[w]["sum"] += real["output"]["blob_output"][w + "_time"]
            timing[w]["min"] = min(
                timing[w]["min"], real["output"]["blob_output"][w + "_time"]
            )
            timing[w]["max"] = max(
                timing[w]["max"], real["output"]["blob_output"][w + "_time"]
            )

    for w in fields:
        timing[w]["average"] = timing[w]["sum"] / len(data)
        print(w)
        for k, v in timing[w].items():
            print(f"\t{k}: {v} seconds")


print("data_fetch.json")
analyze_file(filename="data_fetch.json", fields=["execution", "download"])
print(
    """
real    4m23.058s
user    0m13.354s
sys     0m5.178s
"""
)


print("data_publish.json")
analyze_file(filename="data_publish.json", fields=["execution", "upload"])
print(
    """
real    25m31.582s
user    0m12.525s
sys     0m5.050s
"""
)
