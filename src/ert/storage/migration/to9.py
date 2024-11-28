import json
import os
from pathlib import Path

import polars

info = "Migrate finalized response keys into configs"

def _migrate_response_configs_wrt_finalized_keys(path: Path) -> None:
    for experiment in path.glob("experiments/*"):
        ensembles = path.glob("ensembles/*")

        experiment_id = None
        with open(experiment / "index.json", encoding="utf-8") as f:
            exp_index = json.load(f)
            experiment_id = exp_index["id"]

        responses_config = None
        with open(experiment / "responses.json", mode="r", encoding="utf-8") as f:
            responses_config = json.load(f)
            for response_type, config in responses_config.items():
                if not config.get("has_finalized_keys"):
                    # Read a sample response and write the keys
                    for ens in ensembles:
                        with open(ens / "index.json", encoding="utf-8") as f:
                            ens_file = json.load(f)
                            if ens_file["experiment_id"] != experiment_id:
                                continue

                        real_dirs = [*ens.glob("realization-*")]

                        for real_dir in real_dirs:
                            if (real_dir / f"{response_type}.parquet").exists():
                                df = polars.read_parquet(
                                    real_dir / f"{response_type}.parquet"
                                )
                                response_keys = df["response_key"].unique().to_list()
                                config["has_finalized_keys"] = True
                                config["keys"] = sorted(response_keys)
                                break

                        if config["has_finalized_keys"]:
                            break

                if "has_finalized_keys" not in config:
                    # At this point in "storage history",
                    # only gendata and summary response types
                    # exist, and only summary starts without finalized keys
                    config["has_finalized_keys"] = (
                        config["_ert_kind"] != "SummaryConfig"
                    )

        os.remove(experiment / "responses.json")
        with open(experiment / "responses.json", mode="w+", encoding="utf-8") as f:
            json.dump(
                responses_config,
                f,
                default=str,
                indent=2,
            )


def migrate(path: Path) -> None:
    _migrate_response_configs_wrt_finalized_keys(path)
