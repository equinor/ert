import asyncio
import datetime
import json
import ssl
import sys
from argparse import Namespace
from collections import defaultdict
from dataclasses import dataclass, fields
from functools import partial
from pathlib import Path
from ssl import SSLContext
from typing import Any, assert_never
from urllib.parse import quote

import anyio
import httpx
from resfo_utilities import SummaryKeyType


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
    try:
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
    assert proc.stdout is not None
    while not line.startswith(prefix):
        bline = await proc.stdout.readline()
        line = bline.decode("utf-8")
        await asyncio.sleep(0.01)

    config_path = Path(line[len(prefix) :].strip()) / "storage_server.json"

    async def path_exists() -> None:
        while not config_path.exists():  # noqa: ASYNC110
            await asyncio.sleep(1)

    await asyncio.wait_for(path_exists(), timeout=10)
    return proc, config_path


async def extract_summary_observations(
    url: str,
    token: str,
    ssl_context: ssl.SSLContext,
    config_path: Path,
    experiment: str,
) -> dict[str, list[str]] | None:
    experiments = await get_experiments(url, token, ssl_context)
    experiment_match = next(filter(lambda x: x["id"] == experiment, experiments), None)
    if experiment_match is None:
        raise ValueError(f"Provided experiment id {experiment} not found in storage")
    summary_observations = experiment_match["observations"]["summary"]
    response_key = {v: k for k, vs in summary_observations.items() for v in vs}
    result: dict[str, list[Any]] = defaultdict(list)
    for obs in await get_observations(experiment, config_path):
        if obs["name"] in response_key:
            result[response_key[obs["name"]]].append(obs)
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


async def get_healthy_url(urls: list[str], token: str, ssl_context: SSLContext) -> str:
    for url in urls:
        try:
            async with httpx.AsyncClient(
                base_url=url,
                headers={"Token": token},
                verify=ssl_context,
                timeout=5.0,
            ) as client:
                if (await client.get("healthcheck")).status_code == 200:
                    return url
        except httpx.TransportError:
            pass
    raise RuntimeError(f"Failed to detect a working URL among candidates: {urls}")


async def get_storage_auth(config_path: Path | str) -> tuple[SSLContext, Any]:
    content = await anyio.Path(config_path).read_text()
    storage_config = json.loads(content)

    ssl_context = ssl.create_default_context()
    ssl_context.load_verify_locations(cafile=storage_config["cert"])
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    working_url = await get_healthy_url(
        storage_config["urls"], storage_config["authtoken"], ssl_context
    )
    storage_config["url"] = working_url
    return ssl_context, storage_config


async def get_experiments(
    url: str, token: str, ssl_context: ssl.SSLContext
) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(
        base_url=url,
        headers={"Token": token},
        verify=ssl_context,
    ) as client:
        return json.loads((_ := await client.get("/experiments")).text)


async def get_observations(
    experiment: str, config_path: Path | str
) -> list[dict[str, Any]]:
    ssl_context, storage_config = await get_storage_auth(config_path)

    async with httpx.AsyncClient(
        base_url=storage_config["url"],
        headers={"Token": storage_config["authtoken"]},
        verify=ssl_context,
    ) as client:
        return json.loads(
            (await client.get(f"/experiments/{experiment}/observations")).text
        )


def query_user_for_experiment(url: str, token: str, ssl_context: SSLContext) -> str:
    experiments = asyncio.run(get_experiments(url, token, ssl_context))
    if not experiments:
        print("No experiments found.")
        sys.exit(1)

    print("Available experiments:")
    for idx, exp in enumerate(experiments, 1):
        print(f"  {idx}. {exp['name']} (ID: {exp['id']})")

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
        except (ValueError, KeyboardInterrupt):
            print("\nExiting.")
            sys.exit(1)


def main(args: Namespace, _site_plugins: Any | None = None) -> None:
    proc, config_path = asyncio.run(start_ert_api(args.config))
    assert Path(config_path).is_file()
    ssl_certificate, storage_config = asyncio.run(get_storage_auth(config_path))
    storage_url = storage_config["url"]
    storage_token = storage_config["authtoken"]
    experiment = (
        args.experiment
        if args.experiment is not None
        else query_user_for_experiment(storage_url, storage_token, ssl_certificate)
    )
    try:
        summary_observations = asyncio.run(
            extract_summary_observations(
                storage_url, storage_token, ssl_certificate, config_path, experiment
            )
        )
        if summary_observations is None:
            return
        convert_summary_observations(summary_observations, args.output_csv_file)
    finally:
        proc.terminate()
