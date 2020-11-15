import os
import pathlib
import sys

def _locate_ert_workspace_root(path):
    path = pathlib.Path(path)
    while True:
        if os.path.exists(path/".ert"):
            return path
        if path == pathlib.Path(path.root):
            return None
        path = path.parent


def main():
    if len(sys.argv) == 0:
        return
    elif len(sys.argv) == 1 and sys.argv[0] == "init":
        if _locate_ert_workspace_root(os.getcwd()) is not None:
            sys.exit("Already inside an ERT workspace")

        with open(".ert", "w") as fout:
            fout.write("ERT workspace")
    else:
        sys.exit("Not inside an ERT workspace")
