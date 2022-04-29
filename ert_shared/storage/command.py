import os
from argparse import ArgumentParser


def add_parser_options(ap: ArgumentParser) -> None:
    ap.add_argument(
        "config",
        type=str,
        help=(
            "ERT config file to start the server from "
            "(ignored if --database-url is set)"
        ),
        nargs="?",  # optional
    )
    ap.add_argument(
        "--enable-new-storage",
        action="store_true",
        default=False,
        help="Shorthand for --database-url=sqlite:///ert.db",
    )
    ap.add_argument(
        "--project",
        "-p",
        type=str,
        help="Path to directory in which to create storage_server.json",
        default=os.getcwd(),
    )
    ap.add_argument(
        "--database-url",
        type=str,
        default=None,
        help="Database URL that the server should connect to. (eg. sqlite:///ert.db)",
    )
    ap.add_argument(
        "--host", type=str, default=os.environ.get("ERT_STORAGE_HOST", "127.0.0.1")
    )
    ap.add_argument(
        "--verbose", action="store_true", help="Show verbose output.", default=False
    )
    ap.add_argument("--debug", action="store_true", default=False)
