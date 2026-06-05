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
from httpx import AsyncClient
from natsort import natsorted
from resfo_utilities import SummaryKeyType

from ert.cli.main import ErtCliError


class InvalidSummaryKeyError(ValueError):
    pass


INDENT2 = " " * 2
INDENT4 = " " * 4
INDENT6 = " " * 6


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
        async with asyncio.timeout(60.0):
            while not line.startswith(prefix):
                bline = await proc.stdout.readline()
                line = bline.decode("utf-8")
                await asyncio.sleep(0.01)
    except TimeoutError as e:
        raise ErtCliError("ert api failed to start within 60 seconds") from e

    config_path = Path(line[len(prefix) :].strip()) / "storage_server.json"

    async def path_exists() -> None:
        while not config_path.exists():  # noqa: ASYNC110
            await asyncio.sleep(1)

    await asyncio.wait_for(path_exists(), timeout=10)
    return proc, config_path


async def extract_observations(
    obs_type: str, experiment: dict[str, Any], experiment_id: str, client: AsyncClient
) -> dict[str, Any]:
    if obs_type not in experiment["observations"]:
        return {}
    observations = experiment["observations"][obs_type]
    obs_name_to_summary_key = {
        obs_name: summary_key
        for summary_key, obs_names in observations.items()
        for obs_name in obs_names
    }
    result: dict[str, list[Any]] = defaultdict(list)
    for obs in await get_observations(experiment_id, client):
        if obs["name"] in obs_name_to_summary_key:
            result[obs_name_to_summary_key[obs["name"]]].append(obs)
    return result


def non_empty_fields(skds: list[SummaryKeyData]) -> list[str]:
    if not skds:
        return []
    return [
        f.name
        for f in fields(SummaryKeyData)
        if any(getattr(o, f.name) is not None for o in skds)
    ]


def obs_to_localization_str(obs: list[Any]) -> str | None:
    east = get_first_loc_value("east", obs)
    north = get_first_loc_value("north", obs)
    radius = get_first_loc_value("radius", obs)
    if east is None or north is None:
        return None
    lines = [
        f"{INDENT4}LOCALIZATION {{",
        f"{INDENT6}EAST={east};",
        f"{INDENT6}NORTH={north};",
    ]
    if radius is not None:
        lines.append(f"{INDENT6}RADIUS={radius};")
    lines.append(f"{INDENT4}}};")
    return "\n".join(lines)


def get_first_loc_value(loc_key: str, obs: list[dict[str, Any]]) -> float | int | None:
    for o in obs:
        key_value = o.get(loc_key, [None])[0]
        if key_value is not None:
            return key_value
    return None


def breakthrough_to_string(obs: dict[str, Any], key: str) -> str:
    lines = [
        f"{INDENT4}BREAKTHROUGH {{",
        f"{INDENT6}THRESHOLD={obs['values'][0]};",
        f"{INDENT6}DATE={obs['x_axis'][0]};",
        f"{INDENT6}ERROR={obs['errors'][0]};",
        f"{INDENT6}KEY={key};",
        f"{INDENT4}}};",
    ]
    return "\n".join(lines)


def convert_summary_observations(
    summary_observations: dict[str, list[Any]],
    breakthrough_observations: dict[str, list[Any]],
    csv_file_name: str,
) -> None:
    summary_keys = []
    for key in summary_observations:
        summary_key = make_summary_key_data(key)
        summary_keys.append(summary_key)

    header_fields = non_empty_fields(summary_keys)
    with Path(csv_file_name).open(mode="w", encoding="utf-8") as fout:
        fout.write(", ".join([*header_fields, "value", "error", "date"]))
        fout.write("\n")
        # Sort observations chronologically before natsort
        for obs_list in summary_observations.values():
            obs_list.sort(key=lambda obs: obs["x_axis"][0])
        for key in natsorted(summary_observations.keys()):
            skd = make_summary_key_data(key)
            for observation in summary_observations[key]:
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

    breakthrough = {}
    for brt_key, obs in breakthrough_observations.items():
        key = brt_key.removeprefix("BREAKTHROUGH:")
        summary_key = make_summary_key_data(key)
        if (well := summary_key.well) is not None:
            if len(obs) > 1:
                raise ErtCliError(
                    f"Can only have one breakthrough observation per well.\n"
                    f"Found {len(obs)} breakthroughs for well '{well}'."
                )
            breakthrough[well] = breakthrough_to_string(obs[0], summary_key.keyword)

    localization = {}
    for key in summary_observations | breakthrough_observations:
        summary_key = make_summary_key_data(key.removeprefix("BREAKTHROUGH:"))
        loc_string = obs_to_localization_str(
            summary_observations.get(key, []) + breakthrough_observations.get(key, [])
        )
        if loc_string is not None and (well := summary_key.well) is not None:
            localization[well] = loc_string

    num_obs = sum(len(obs_list) for obs_list in summary_observations.values()) + (
        1 if breakthrough else 0
    )
    obs_names = [
        obs_dict["name"]
        for obs_list in summary_observations.values()
        for obs_dict in obs_list
    ] + [
        obs_dict["name"]
        for obs_list in breakthrough_observations.values()
        for obs_dict in obs_list
    ]
    obs_names_str = INDENT4 + f"\n{INDENT4}".join(obs_names)
    bulk_config_list = ["SUMMARY {", f"{INDENT2}VALUES = {csv_file_name};"]
    for well in sorted(localization.keys() | breakthrough.keys()):
        bulk_config_list.extend(
            [
                f"{INDENT2}WELL {well} {{",
                localization.get(well, ""),
                breakthrough.get(well, ""),
                f"{INDENT2}}};",
            ]
        )
    bulk_config_list.append("};")
    bulk_config_str = "\n".join(
        filter(None, bulk_config_list)  # Filter out empty (falsy) strings
    )

    print(
        f"\n{num_obs} observations can be replaced by: \n"
        f"{INDENT2}1.  Copying the file '{csv_file_name}' to the folder containing "
        f"your observation  configuration.\n"
        f"{INDENT2}2.  Replacing the named observations below with the bulk "
        f"configuration\n\n"
        f"Observation names (to replace):\n"
        f"==============================\n"
        f"{obs_names_str}\n\n"
        f"Bulk configuration (replace with):\n"
        f"=================================\n"
        f"{bulk_config_str}"
    )


async def get_storage_auth(config_path: Path | str) -> tuple[SSLContext, Any]:
    content = await anyio.Path(config_path).read_text()
    storage_config = json.loads(content)

    ssl_context = ssl.create_default_context()
    ssl_context.load_verify_locations(cafile=storage_config["cert"])
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    return ssl_context, storage_config


async def fetch_experiments(
    client: httpx.AsyncClient,
) -> list[dict[str, Any]]:
    all_experiments = json.loads((_ := await client.get("/experiments")).text)
    if not isinstance(all_experiments, list):
        raise ErtCliError("Could not fetch experiments from storage.")
    if len(all_experiments) == 0:
        raise ErtCliError("Could not find any experiments in storage.")
    return all_experiments


async def get_observations(
    experiment: str, client: httpx.AsyncClient
) -> list[dict[str, Any]]:
    return json.loads(
        (await client.get(f"/experiments/{experiment}/observations")).text
    )


def query_user_for_experiment(experiments: list[dict[str, Any]]) -> str:
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
    urls: list[str], token: str, ssl_context: SSLContext, retries: int = 3
) -> AsyncGenerator[httpx.AsyncClient, None]:
    for attempt in range(retries):
        for url in urls:
            with suppress(httpx.TransportError):
                async with httpx.AsyncClient(
                    base_url=url,
                    headers={"Token": token},
                    verify=ssl_context,
                    timeout=5.0,
                ) as client:
                    if (await client.get("healthcheck")).status_code == 200:
                        yield client
                        return
        if attempt < retries - 1:
            await asyncio.sleep(1)
    raise ErtCliError(
        f"Failed to detect a working storage URL after {retries} retries."
    )


async def _run_with_client(
    ssl_certificate: SSLContext, storage_config: Any, args: Namespace
) -> None:
    async with connect(
        storage_config["urls"], storage_config["authtoken"], ssl_certificate
    ) as client:
        all_experiments = await fetch_experiments(client)
        assert isinstance(all_experiments, list)
        experiment_id = get_experiment_id(all_experiments, args.experiment_id)
        experiment = get_experiment(all_experiments, experiment_id)
        summary_observations = await extract_observations(
            "summary", experiment, experiment_id, client
        )
        if not summary_observations:
            raise ErtCliError(
                f"No summary observations found for experiment '{experiment_id}'"
            )
        breakthrough_observations = await extract_observations(
            "breakthrough", experiment, experiment_id, client
        )

        convert_summary_observations(
            summary_observations, breakthrough_observations, args.output_csv_file
        )


def get_experiment(
    all_experiments: list[dict[str, Any]], experiment_id: str
) -> dict[str, Any]:
    experiment = next((ex for ex in all_experiments if ex["id"] == experiment_id), None)
    if experiment is None:
        raise ErtCliError(f"No experiment with id {experiment_id} found in storage")
    return experiment


def get_experiment_id(
    all_experiments: list[dict[str, Any]], arg_experiment_id: str | None
) -> str:
    if arg_experiment_id is not None:
        experiment_id = arg_experiment_id
        experiment_ids = [ex["id"] for ex in all_experiments]
        if experiment_id not in experiment_ids:
            raise ErtCliError(
                f"An experiment with id '{experiment_id}' does not exist.\n"
                f"Available experiments:\n  "
                + "\n  ".join(
                    [
                        f"{i}. {ex['id']} ({ex['name']})"
                        for i, ex in enumerate(all_experiments, 1)
                    ]
                )
            )
    elif len(all_experiments) == 1:
        experiment_id = all_experiments[0]["id"]
        print(
            f"Gathering observations for sole experiment in storage: '{experiment_id}'"
        )
    else:
        experiment_id = query_user_for_experiment(all_experiments)

    return experiment_id


async def _async_main(args: Namespace) -> None:
    proc = None
    try:
        proc, config_path = await start_ert_api(args.config)
        ssl_certificate, storage_config = await get_storage_auth(config_path)
        await _run_with_client(ssl_certificate, storage_config, args)
    finally:
        if proc is not None and proc.returncode is None:
            proc.terminate()
            await proc.wait()


def main(args: Namespace, _site_plugins: Any | None = None) -> None:
    asyncio.run(_async_main(args))
