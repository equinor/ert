#!/usr/bin/env python

import json
import requests
import os
import sys
import subprocess


try:
    ERT_STORAGE_URL = os.environ["ERT_STORAGE_URL"]
except KeyError:
    ERT_STORAGE_URL = "http://127.0.0.1:8000"


def get_ensemble_id():
    resp = requests.post(f"{ERT_STORAGE_URL}/ensembles", json={"parameters": []})
    return resp.json()["id"]


def create_realization_record_file(realizations):
    if os.path.exists("iens.json"):
        os.remove("iens.json")

    with open("iens.json", "w") as f:
        f.write(json.dumps([{"iens": x + 1} for x in range(realizations)]))


# 100 reals 1 gig
if __name__ == "__main__":
    nr_realizations = eval(sys.argv[1])
    create_realization_record_file(nr_realizations)
    ens_id = get_ensemble_id()
    subprocess.run(["sed", "-i", f"s/ens_id/{ens_id}/g", "stages.yml"])
    subprocess.run(
        ["sed", "-i", f"s/10/{nr_realizations}/g", "experiments/publish/ensemble.yml"]
    )
    subprocess.run(
        ["sed", "-i", f"s/10/{nr_realizations}/g", "experiments/fetch/ensemble.yml"]
    )
    subprocess.run(
        [
            "sed",
            "-i",
            f"s/10/{nr_realizations}/g",
            "experiments/publish_direct/ensemble.yml",
        ]
    )
