# Consume a text file with a design matrix emitted by `design2params` [semeio]
# and writes json files adapted for loading as ensemble records.

import json
from pathlib import Path

import pandas as pd

DESIGN2PARAMS_OUTPUT = "designmatrix.txt"


if __name__ == "__main__":
    dframe = pd.read_csv(DESIGN2PARAMS_OUTPUT, sep=" ")
    for col in dframe.columns[1:]:
        Path("resources/designed_" + col.lower() + ".json").write_text(
            json.dumps(list(dframe[col]), indent=4), encoding="utf-8"
        )
