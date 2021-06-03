import os


def add_parser_options(ap):
    ap.add_argument(
        "--runpath",
        type=str,
        help="Path to where the database-files are located.",
        default=os.getcwd(),
    )
    ap.add_argument(
        "--verbose", action="store_true", help="Show verbose output.", default=False
    )
    ap.add_argument(
        "--disable-lockfile",
        action="store_true",
        default=False,
        help="Don't create storage_server.json",
    )
    ap.add_argument(
        "--rdb-url", type=str, default=f"sqlite:///{os.getcwd()}/ert_storage.db"
    )
    ap.add_argument(
        "--host", type=str, default=os.environ.get("ERT_STORAGE_HOST", "127.0.0.1")
    )
    ap.add_argument("--debug", action="store_true", default=False)
