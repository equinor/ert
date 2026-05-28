import asyncio
import datetime
import json
import ssl
from argparse import Namespace
from collections import defaultdict
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, fields
from functools import partial
from pathlib import Path
from ssl import SSLContext
from typing import Any, assert_never
from urllib.parse import quote

import anyio
import httpx
from resfo_utilities import SummaryKeyType

from ert.cli.main import ErtCliError


class InvalidSummaryKeyError(ValueError):
    pass


@dataclass
class SummaryKeyData:
    keyword: str
    number: int | None = None
    well: str | None = None
    name: str | None = None
    i: int | None = None
    j: int | None = None
    k: int | None = None
    lgr_name: str | None = None
    region1: int | None = None
    region2: int | None = None


def make_summary_key_data(summary_key: str) -> SummaryKeyData:
    fields = summary_key.split(":")
    summary_variable = fields[0]
    skd = partial(SummaryKeyData, keyword=summary_variable)
    try:  # noqa: PLW0717
        match SummaryKeyType.from_variable(summary_variable):
            case SummaryKeyType.FIELD | SummaryKeyType.OTHER:
                return skd()
            case (
                SummaryKeyType.REGION | SummaryKeyType.AQUIFER | SummaryKeyType.NETWORK
            ):
                return skd(number=int(fields[1]))
            case SummaryKeyType.BLOCK:
                i, j, k = fields[1].split(",")
                return skd(i=int(i), j=int(j), k=int(k))
            case SummaryKeyType.WELL:
                return skd(well=fields[1])
            case SummaryKeyType.GROUP:
                return skd(name=fields[1])
            case SummaryKeyType.SEGMENT:
                return skd(name=fields[1], number=int(fields[2]))
            case SummaryKeyType.COMPLETION:
                i, j, k = fields[2].split(",")
                return skd(name=fields[1], i=int(i), j=int(j), k=int(k))
            case SummaryKeyType.INTER_REGION:
                r1, r2 = fields[1].split("-")
                return skd(region1=int(r1), region2=int(r2))
            case SummaryKeyType.LOCAL_WELL:
                return skd(lgr_name=fields[1], name=fields[2])
            case SummaryKeyType.LOCAL_BLOCK:
                i, j, k = fields[2].split(",")
                return skd(lgr_name=fields[1], i=int(i), j=int(j), k=int(k))
            case SummaryKeyType.LOCAL_COMPLETION:
                i, j, k = fields[3].split(",")
                return skd(
                    lgr_name=fields[1], name=fields[2], i=int(i), j=int(j), k=int(k)
                )
            case default:
                assert_never(default)
    except Exception as err:
        raise InvalidSummaryKeyError(f"Invalid summary key {summary_key}") from err


def escape(s: str) -> str:
    return quote(s, safe="")


async def start_ert_api(ert_config: str) -> tuple[asyncio.subprocess.Process, Path]:
    proc = await asyncio.create_subprocess_exec(
        "ert",
        "api",
        ert_config,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    line = ""
    prefix = "Serving storage directory:"
    print("Waiting for storage server ...")
    assert proc.stdout is not None
    try:
        async with asyncio.timeout(30.0):
            while not line.startswith(prefix):
                bline = await proc.stdout.readline()
                line = bline.decode("utf-8")
                await asyncio.sleep(0.01)
    except TimeoutError as e:
        raise ErtCliError("ert api failed to start within 30 seconds") from e

    config_path = Path(line[len(prefix) :].strip()) / "storage_server.json"

    async def path_exists() -> None:
        while not config_path.exists():  # noqa: ASYNC110
            await asyncio.sleep(1)

    await asyncio.wait_for(path_exists(), timeout=10)
    return proc, config_path


async def extract_summary_observations(
    client: httpx.AsyncClient,
    experiment: str,
    experiments: list[Any],
) -> dict[str, list[Any]]:
    experiment_match = next(filter(lambda x: x["id"] == experiment, experiments), None)
    if experiment_match is None:
        raise ErtCliError(f"Provided experiment id {experiment} not found in storage")
    if "summary" not in experiment_match["observations"]:
        raise ErtCliError(
            f"No summary observations found for experiment '{experiment}'"
        )
    summary_observations = experiment_match["observations"]["summary"]
    response_key = {v: k for k, vs in summary_observations.items() for v in vs}
    result: dict[str, list[Any]] = defaultdict(list)
    for obs in await get_observations(experiment, client):
        if obs["name"] in response_key:
            result[response_key[obs["name"]]].append(obs)
    if len(result) == 0:
        raise ErtCliError(
            f"No summary observations found for experiment '{experiment}'"
        )
    return result


def non_empty_fields(skds: list[SummaryKeyData]) -> list[str]:
    if not skds:
        return []
    return [
        f.name
        for f in fields(SummaryKeyData)
        if any(getattr(o, f.name) is not None for o in skds)
    ]


def convert_summary_observations(
    summary_observations: dict[str, list[Any]], csv_file_name: str
) -> None:
    print(f"SUMMARY {{\n  VALUES = {csv_file_name};\n}};\n")
    skds = [make_summary_key_data(key) for key in summary_observations]
    header_fields = non_empty_fields(skds)
    with Path(csv_file_name).open(mode="w", encoding="utf-8") as fout:
        fout.write(", ".join([*header_fields, "value", "error", "date"]))
        fout.write("\n")
        for key, observations in summary_observations.items():
            skd = make_summary_key_data(key)
            for observation in observations:
                for f in header_fields:
                    v = getattr(skd, f)
                    if v is None:
                        fout.write(", ")
                    else:
                        fout.write(f"{v}, ")
                date = datetime.datetime.fromisoformat(observation["x_axis"][0])
                fout.write(f"{observation['values'][0]:.3g}, ")
                fout.write(f"{observation['errors'][0]:.3g}, ")
                fout.write(f"{date.isoformat()}\n")


async def get_storage_auth(config_path: Path | str) -> tuple[SSLContext, Any]:
    content = await anyio.Path(config_path).read_text()
    storage_config = json.loads(content)

    ssl_context = ssl.create_default_context()
    ssl_context.load_verify_locations(cafile=storage_config["cert"])
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    return ssl_context, storage_config


async def get_experiments(
    client: httpx.AsyncClient,
) -> list[dict[str, Any]]:
    all_experiments = json.loads((_ := await client.get("/experiments")).text)
    if (
        isinstance(all_experiments, dict)
        and all_experiments.get("detail", {}).get("error") is not None
    ):
        raise ErtCliError("Could not find any experiments in storage.")
    return all_experiments


async def get_observations(
    experiment: str, client: httpx.AsyncClient
) -> list[dict[str, Any]]:
    return json.loads(
        (await client.get(f"/experiments/{experiment}/observations")).text
    )


def query_user_for_experiment(experiments: list[dict[str, Any]]) -> str:
    if len(experiments) == 1:
        print(
            f"Only one experiment found, "
            f"picking experiment with id '{experiments[0]['id']}'"
        )
        return experiments[0]["id"]
    print("Available experiments:")
    for idx, exp in enumerate(experiments, 1):
        print(f"  {idx}. {exp['id']} ({exp['name']})")

    while True:
        try:
            choice = input("\nSelect an experiment (enter number): ").strip()
            selected_idx = int(choice) - 1
            if 0 <= selected_idx < len(experiments):
                return experiments[selected_idx]["id"]
            print(
                f"Invalid choice. "
                f"Please enter a number between 1 and {len(experiments)}"
            )
        except (ValueError, KeyboardInterrupt) as e:
            raise ErtCliError("No experiment selected, exiting ...") from e


@asynccontextmanager
async def connect(
    urls: list[str], token: str, ssl_context: SSLContext
) -> AsyncGenerator[httpx.AsyncClient, None]:
    found_url = False
    for url in urls:
        if found_url:
            continue
        with suppress(httpx.TransportError):
            async with httpx.AsyncClient(
                base_url=url,
                headers={"Token": token},
                verify=ssl_context,
                timeout=5.0,
            ) as client:
                if (await client.get("healthcheck")).status_code == 200:
                    found_url = True
                    yield client
    if not found_url:
        raise ErtCliError(f"Failed to detect a working URL among candidates: {urls}")


async def _run_with_client(
    ssl_certificate: SSLContext, storage_config: Any, args: Namespace
) -> None:
    async with connect(
        storage_config["urls"], storage_config["authtoken"], ssl_certificate
    ) as client:
        all_experiments = await get_experiments(client)

        assert isinstance(all_experiments, list)
        if args.experiment is not None:
            experiment = args.experiment
            experiment_ids = [ex["id"] for ex in all_experiments]
            if experiment not in experiment_ids:
                raise ErtCliError(
                    f"An experiment with id '{experiment}' does not exist.\n"
                    f"Available experiments:\n  "
                    + "\n  ".join(
                        [
                            f"{i}. {ex['id']} ({ex['name']})"
                            for i, ex in enumerate(all_experiments)
                        ]
                    )
                )
        else:
            experiment = query_user_for_experiment(all_experiments)

        summary_observations = await extract_summary_observations(
            client, experiment, all_experiments
        )

        convert_summary_observations(summary_observations, args.output_csv_file)


def main(args: Namespace, _site_plugins: Any | None = None) -> None:
    proc = None
    try:
        proc, config_path = asyncio.run(start_ert_api(args.config))
        ssl_certificate, storage_config = asyncio.run(get_storage_auth(config_path))
        asyncio.run(_run_with_client(ssl_certificate, storage_config, args))
    finally:
        if proc is not None:
            proc.terminate()
