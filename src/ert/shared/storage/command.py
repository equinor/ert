import os
from argparse import ArgumentParser


def add_parser_options(ap: ArgumentParser) -> None:
    ap.add_argument(
        "config",
        type=str,
        help=("ERT config file to start the server from "),
        nargs="?",  # optional
    )
    ap.add_argument(
        "--project",
        "-p",
        type=str,
        help="Path to directory in which to create storage_server.json",
        default=os.getcwd(),
    )
    ap.add_argument(
        "--traceparent",
        type=str,
        help="Trace parent id to be used by the storage root span",
        default=None,
    )
    ap.add_argument(
        "--parent_pid",
        type=int,
        help="The parent process id",
        default=os.getppid(),
    )
    ap.add_argument(
        "--host", type=str, default=os.environ.get("ERT_STORAGE_HOST", "127.0.0.1")
    )
    ap.add_argument("--logging-config", type=str, default=None)
    ap.add_argument(
        "--verbose", action="store_true", help="Show verbose output.", default=False
    )
    ap.add_argument("--debug", action="store_true", default=False)
