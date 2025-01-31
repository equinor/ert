import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import polars as pl

info = "Migrate finalized response keys into configs"


def _write_transaction(filename: str | os.PathLike[str], data: bytes) -> None:
    """
    Writes the data to the filename as a transaction.

    Guarantees to not leave half-written or empty files on disk if the write
    fails or the process is killed.
    """

    Path("./swp").mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir="./swp", delete=False) as f:
        f.write(data)
        os.chmod(f.name, 0o660)
        os.rename(f.name, filename)


def migrate(path: Path) -> None:
    for experiment in path.glob("experiments/*"):
        ensembles = [*path.glob("ensembles/*")]

        with (
            open(experiment / "index.json", encoding="utf-8") as f_experiment,
            open(experiment / "responses.json", encoding="utf-8") as f_responses,
        ):
            exp_index = json.load(f_experiment)
            experiment_id = exp_index["id"]

            responses_config = json.load(f_responses)
            for response_type, config in responses_config.items():
                if not config.get("has_finalized_keys"):
                    # Read a sample response and write the keys
                    for ens in ensembles:
                        with open(ens / "index.json", encoding="utf-8") as f_ensemble:
                            ens_file = json.load(f_ensemble)
                            if ens_file["experiment_id"] != experiment_id:
                                continue

                        real_dirs = [*ens.glob("realization-*")]

                        for real_dir in real_dirs:
                            if (real_dir / f"{response_type}.parquet").exists():
                                df = pl.scan_parquet(
                                    real_dir / f"{response_type}.parquet"
                                )
                                response_keys = (
                                    df.select("response_key")
                                    .unique()
                                    .collect()
                                    .to_series()
                                    .to_list()
                                )
                                config["has_finalized_keys"] = True
                                config["keys"] = sorted(response_keys)
                                break

                        if config.get("has_finalized_keys"):
                            break

                # If this is hit, it means no responses were found
                # for that response type, so we still cannot have "finalized"
                # keys. We then default it to that of the configs.
                # At time of writing, it is always True for GenDataConfig
                # and False for SummaryConfig
                if "has_finalized_keys" not in config:
                    config["has_finalized_keys"] = (
                        config["_ert_kind"] != "SummaryConfig"
                    )

        _write_transaction(
            experiment / "responses.json",
            json.dumps(
                responses_config,
                default=str,
                indent=2,
            ).encode("utf-8"),
        )
