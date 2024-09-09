import json
import os

import everest
import pytest
from ruamel.yaml import YAML

from tests.everest.utils import tmpdir


@tmpdir(None)
def test_safe_open():
    filename = "a/relative/path"
    assert not os.path.exists(filename)

    with pytest.raises(IOError), everest.jobs.io.safe_open(filename) as fout:
        print(fout.readlines())

    with everest.jobs.io.safe_open(filename, "w") as fout:
        fout.write("testing testing")

    assert os.path.exists(filename)
    with everest.jobs.io.safe_open(filename) as fin:
        assert "".join(fin.readlines()) == "testing testing"


@tmpdir(None)
def test_load_data():
    data = {"secret": "data"}

    json_file = "data.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f)
    assert data == everest.jobs.io.load_data(json_file)

    yml_file = "data.yml"
    with open(yml_file, "w", encoding="utf-8") as f:
        YAML(typ="safe", pure=True).dump(data, f)
    assert data == everest.jobs.io.load_data(yml_file)

    garbage_file = "garbage.yml"
    with open(garbage_file, "w", encoding="utf-8") as f:
        f.write("[")

    with pytest.raises(IOError):
        everest.jobs.io.load_data(garbage_file)
