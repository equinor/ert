#!/usr/bin/env python
import os
import sys
import json
import requests
import hashlib
import time

# "s034-0113.s034.oc.equinor.com:8000"
try:
    ERT_STORAGE_URL = os.environ["ERT_STORAGE_URL"]
except KeyError:
    ERT_STORAGE_URL = "http://127.0.0.1:8000"

RECORD_NAMES = {"published_file", "published_matrix"}


def get_record(record_name, ensemble_id, realization_index):
    resp = requests.get(
        f"{ERT_STORAGE_URL}/ensembles/{ensemble_id}/records/{record_name}?realization_index={realization_index}"
    )
    assert resp.status_code == 200
    return resp


def main():
    start = time.time()
    with open("realization.json") as f:
        realization = json.loads(f.read())
    ensemble_id = sys.argv[1]
    realization_index = int(realization["iens"])
    download_times = {
        "sum": 1,
    }
    for record in RECORD_NAMES:
        start_download = time.time()
        data = get_record(record, ensemble_id, realization_index).content
        download_time = time.time() - start_download
        download_times[f"{record}_time"] = download_time
        md5 = hashlib.md5()
        md5.update(data)
        _hash = md5.hexdigest()  # str
        checksum = get_record(f"{record}_checksum", ensemble_id, realization_index).text
        assert _hash == checksum

    duration = time.time() - start
    download_times["execution_time"] = duration
    with open("output.json", "w") as f:
        f.write(json.dumps(download_times))


if __name__ == "__main__":
    main()
