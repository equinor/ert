# Consume a text file with a design matrix emitted by `design2params` [semeio]
# and writes json files adapted for loading as ensemble records.

import json
from pathlib import Path

import pandas as pd

DESIGN2PARAMS_OUTPUT = "designmatrix.txt"


if __name__ == "__main__":
    dframe = pd.read_csv(DESIGN2PARAMS_OUTPUT, sep=" ")
    for col in dframe.columns[1:]:
        ens_records = [[value] for value in dframe[col]]
        Path("resources/designed_" + col.lower() + ".json").write_text(
            json.dumps(ens_records, indent=4), encoding="utf-8"
        )

    # This file is to be used when #1437 is resolved
    Path("resources/designed_futureuse.json").write_text(
        dframe.to_json(orient="records", indent=4), encoding="utf-8"
    )
