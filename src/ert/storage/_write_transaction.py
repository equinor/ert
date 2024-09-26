from __future__ import annotations

import os


def write_transaction(filename: str | os.PathLike[str], data: bytes) -> None:
    """Writes the data to the filename as a transaction.

    Guarantees to not leave half-written or empty files on disk if the write
    fails or the process is killed.
    """
    swapfile = str(filename) + ".swp"
    with open(swapfile, mode="wb") as f:
        f.write(data)

    os.rename(swapfile, filename)
