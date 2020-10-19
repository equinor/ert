import sys
from argparse import ArgumentParser

from ert_shared.storage.command import add_parser_options
from ert_shared.storage.http_server import run_server


if __name__ == "__main__":
    ap = ArgumentParser()
    add_parser_options(ap)

    args = ap.parse_args()
    run_server(args)
