import json
import os

from ruamel.yaml import YAML


def load_data(filename):
    """Will try to load data from @filename first as yaml, and if that fails,
    as json. If both fail, a ValueError with both of the error messages will be
    raised.
    """
    yaml_err = ""
    json_err = ""
    with open(filename, encoding="utf-8") as fin:
        try:
            return YAML(typ="safe", pure=True).load(fin)
        except Exception as err:
            yaml_err = err

        try:
            return json.load(fin)
        except Exception as err:
            json_err = err

    err_msg = "%s is neither yaml (err_msg=%s) nor json (err_msg=%s)"
    raise IOError(err_msg % (filename, str(yaml_err), str(json_err)))


def _create_folders(filename):
    dirname = os.path.realpath(os.path.dirname(filename))
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def safe_open(file, mode="r", **kwargs):
    """If @file is opened in a writable mode (mode contains 'w', 'x', 'a', or
    '+') it first creates all potentially missing directories, before opening
    the file
    """

    if any(wtoken in mode for wtoken in "wxa+"):
        _create_folders(file)
    return open(file, mode, encoding="utf-8", **kwargs)  # noqa SIM115
