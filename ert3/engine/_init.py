import ert3

from pathlib import Path
import sys


def init_workspace(path):
    path = Path(path)
    if ert3._locate_ert_workspace_root(path) is not None:
        sys.exit("Already inside an ERT workspace")

    ert3.storage.init(path)
