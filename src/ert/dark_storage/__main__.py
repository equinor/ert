"""
Start a debug uvicorn server
"""
import os
import sys
import uvicorn
from typing import List, Optional
from tempfile import mkdtemp
from shutil import rmtree


def run_server() -> None:
    database_dir: Optional[str] = None
    if "ERT_STORAGE_DATABASE_URL" not in os.environ:
        print(
            "Environment variable 'ERT_STORAGE_DATABASE_URL' not set.\n"
            "Defaulting to development SQLite temporary database.\n"
            "Configure:\n"
            "1. File-based SQLite (development):\n"
            "\tERT_STORAGE_DATABASE_URL=sqlite:///ert.db\n"
            "2. PostgreSQL (production):\n"
            "\tERT_STORAGE_DATABASE_URL=postgresql:///<username>:<password>@<hostname>:<port>/<database>\n",
            file=sys.stderr,
        )
        database_dir = mkdtemp(prefix="ert-storage_")
        os.environ["ERT_STORAGE_DATABASE_URL"] = f"sqlite:///{database_dir}/ert.db"

    if "ERT_STORAGE_AZURE_CONNECTION_STRING" not in os.environ:
        print(
            "Environment variable 'ERT_STORAGE_AZURE_CONNECTION_STRING' not set.\n"
            "Not using Azure Blob Storage. Blob data will be stored in the RDBMS.\n"
        )

    try:
        uvicorn.run(
            "ert_storage.app:app", reload=True, reload_dirs=[os.path.dirname(__file__)]
        )
    finally:
        if database_dir is not None:
            rmtree(database_dir)


def run_alembic(args: List[str]) -> None:
    """
    Forward arguments to alembic
    """
    from alembic.config import main as alembic_main

    dbkey = "ERT_STORAGE_DATABASE_URL"
    dburl = os.getenv(dbkey)
    if dburl is None:
        sys.exit(
            f"Environment variable '{dbkey}' not set.\n"
            "It needs to point to a PostgreSQL server for alembic to work."
        )
    if not dburl.startswith("postgresql"):
        sys.exit(
            f"Environment variable '{dbkey}' does not point to a postgresql database.\n"
            "Only PostgreSQL is supported for alembic migrations at the moment.\n"
            f"Its value is: {dburl}"
        )

    argv = [
        "-c",
        os.path.join(os.path.dirname(__file__), "_alembic", "alembic.ini"),
        *args,
    ]

    try:
        alembic_main(argv=argv, prog="ert-storage alembic")
    except FileNotFoundError as exc:
        if os.path.basename(exc.filename) == "script.py.mako":
            sys.exit(
                f"\nAlembic could not find 'script.py.mako' in location:\n"
                f"\n{exc.filename}\n\n"
                "This is most likely because you've installed ert-storage without --edit mode\n"
                "Reinstall ert-storage with: pip install -e <path to ert-storage>"
            )
        else:
            raise
    sys.exit(0)


def print_usage() -> None:
    sys.exit(
        "Usage: ert-storage [alembic...]\n\n"
        "If alembic is given as the first argument, forward the rest of the\n"
        "arguments to alembic. Otherwise start ERT Storage in development mode."
    )


def main(args: Optional[List[str]] = None) -> None:
    if args is None:
        args = sys.argv[1:]

    if len(args) > 0:
        if args[0] == "alembic":
            run_alembic(args[1:])
        else:
            print_usage()
    run_server()


if __name__ == "__main__":
    main()
