import asyncio
import json
import ssl
from collections import defaultdict
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from ssl import SSLContext
from typing import Any, Literal
from urllib.parse import quote

import anyio
import httpx
from httpx import AsyncClient
from natsort import natsorted

from ert.cli.main import ErtCliError
from ert.config import (
    SummaryKeyData,
    make_summary_key_data,
)
from ert.namespace import Namespace

INDENT2 = " " * 2
INDENT4 = " " * 4
INDENT6 = " " * 6


def _get_first_loc_value(loc_key: str, obs: list[dict[str, Any]]) -> float | int | None:
    for o in obs:
        key_value = o.get(loc_key, [None])[0]
        if key_value is not None:
            return key_value
    return None


def _obs_to_localization_str(obs: list[Any]) -> str | None:
    east = _get_first_loc_value("east", obs)
    north = _get_first_loc_value("north", obs)
    radius = _get_first_loc_value("radius", obs)
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


def _non_empty_fields(skds: list[SummaryKeyData]) -> list[str]:
    if not skds:
        return []
    return [
        f.name
        for f in fields(SummaryKeyData)
        if any(getattr(o, f.name) is not None for o in skds)
    ]


def _breakthrough_to_string(obs: dict[str, Any], key: str) -> str:
    lines = [
        f"{INDENT4}BREAKTHROUGH {{",
        f"{INDENT6}THRESHOLD={obs['values'][0]};",
        f"{INDENT6}DATE={obs['x_axis'][0]};",
        f"{INDENT6}ERROR={obs['errors'][0]};",
        f"{INDENT6}KEY={key};",
        f"{INDENT4}}};",
    ]
    return "\n".join(lines)


class BulkConfigExporter:
    def __init__(
        self,
        summary_observations: dict[str, Any],
        breakthrough_observations: dict[str, Any],
    ) -> None:
        self.summary_observations = summary_observations
        self.breakthrough_observations = breakthrough_observations
        self.csv_file_name = "summary_observations.csv"

        self.well_to_breakthrough = self._map_well_to_breakthrough()
        self.well_to_localization = self._map_localization_to_well()

        summary_keys = []
        for key in summary_observations:
            summary_key = make_summary_key_data(key)
            summary_keys.append(summary_key)

        self.header_fields = _non_empty_fields(summary_keys)

    def _map_localization_to_well(self) -> dict[str, Any]:
        well_to_localization = {}
        for key in self.summary_observations | self.breakthrough_observations:
            summary_key = make_summary_key_data(key.removeprefix("BREAKTHROUGH:"))
            loc_string = _obs_to_localization_str(
                self.summary_observations.get(key, [])
                + self.breakthrough_observations.get(key, [])
            )
            if loc_string is not None and (well := summary_key.well) is not None:
                well_to_localization[well] = loc_string
        return well_to_localization

    def _map_well_to_breakthrough(self) -> dict[str, Any]:
        well_to_breakthrough = {}
        for brt_key, obs in self.breakthrough_observations.items():
            key = brt_key.removeprefix("BREAKTHROUGH:")
            summary_key = make_summary_key_data(key)
            if (well := summary_key.well) is not None:
                if len(obs) > 1:
                    raise ErtCliError(
                        f"Can only have one breakthrough observation per well.\n"
                        f"Found {len(obs)} breakthroughs for well '{well}'."
                    )
                well_to_breakthrough[well] = _breakthrough_to_string(
                    obs[0], summary_key.keyword
                )
        return well_to_breakthrough

    def write_csv(self) -> None:
        with Path(self.csv_file_name).open(mode="w", encoding="utf-8") as fout:
            fout.write(", ".join([*self.header_fields, "value", "error", "date"]))
            fout.write("\n")
            # Sort observations chronologically before natsort
            for obs_list in self.summary_observations.values():
                obs_list.sort(key=lambda obs: obs["x_axis"][0])
            for key in natsorted(self.summary_observations.keys()):
                skd = make_summary_key_data(key)
                for observation in self.summary_observations[key]:
                    for f in self.header_fields:
                        v = getattr(skd, f)
                        if v is None:
                            fout.write(", ")
                        else:
                            fout.write(f"{v}, ")
                    date = datetime.fromisoformat(observation["x_axis"][0])
                    fout.write(f"{observation['values'][0]:.3g}, ")
                    fout.write(f"{observation['errors'][0]:.3g}, ")
                    fout.write(f"{date.isoformat()}\n")

    def print_bulk_config(
        self,
    ) -> None:
        obs_names = [
            obs_dict["name"]
            for obs_list in self.summary_observations.values()
            for obs_dict in obs_list
        ] + [
            obs_dict["name"]
            for obs_list in self.breakthrough_observations.values()
            for obs_dict in obs_list
        ]
        obs_names_str = INDENT4 + f"\n{INDENT4}".join(obs_names)
        bulk_config_list = ["SUMMARY {", f"{INDENT2}VALUES = {self.csv_file_name};"]
        for well in sorted(
            self.well_to_localization.keys() | self.well_to_breakthrough.keys()
        ):
            bulk_config_list.extend(
                [
                    f"{INDENT2}WELL {well} {{",
                    self.well_to_localization.get(well, ""),
                    self.well_to_breakthrough.get(well, ""),
                    f"{INDENT2}}};",
                ]
            )
        bulk_config_list.append("};")
        bulk_config_str = "\n".join(
            filter(None, bulk_config_list)
            # Filter out empty (falsy) strings
        )
        print(
            f"\n{len(obs_names)} observations can be replaced by: \n"
            f"{INDENT2}1.  Copying the file '{self.csv_file_name}' to the folder "
            f"containing your observation configuration.\n"
            f"{INDENT2}2.  Replacing the named observations below with the bulk "
            f"configuration\n\n"
            f"Observation names (to replace):\n"
            f"==============================\n"
            f"{obs_names_str}\n\n"
            f"Bulk configuration (replace with):\n"
            f"=================================\n"
            f"{bulk_config_str}"
        )


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


ObsDict = dict[Literal["summary", "breakthrough"], Any]


async def collect_all_observations(client: AsyncClient) -> ObsDict:
    observations: ObsDict = {}
    all_experiments = await fetch_experiments(client)
    experiment_id = get_experiment_id(all_experiments)
    experiment = get_experiment(all_experiments, experiment_id)
    observations["summary"] = await extract_observations(
        "summary", experiment, experiment_id, client
    )
    if not observations["summary"]:
        raise ErtCliError(
            f"No summary observations found for experiment '{experiment_id}'"
        )
    observations["breakthrough"] = await extract_observations(
        "breakthrough", experiment, experiment_id, client
    )

    return observations


def get_experiment(
    all_experiments: list[dict[str, Any]], experiment_id: str
) -> dict[str, Any]:
    experiment = next((ex for ex in all_experiments if ex["id"] == experiment_id), None)
    if experiment is None:
        raise ErtCliError(f"No experiment with id {experiment_id} found in storage")
    return experiment


def get_experiment_id(all_experiments: list[dict[str, Any]]) -> str:
    if len(all_experiments) == 1:
        experiment_id = all_experiments[0]["id"]
        print(
            f"Gathering observations for sole experiment in storage: '{experiment_id}'"
        )
    else:
        experiment_id = query_user_for_experiment(all_experiments)

    return experiment_id


async def _convert_summary_to_bulk_config(config: str) -> None:
    proc = None
    try:
        proc, config_path = await start_ert_api(config)
        ssl_certificate, storage_config = await get_storage_auth(config_path)
        async with connect(
            storage_config["urls"], storage_config["authtoken"], ssl_certificate
        ) as client:
            observations = await collect_all_observations(client)
    finally:
        if proc is not None and proc.returncode is None:
            proc.terminate()
            await proc.wait()

    bulk_exporter = BulkConfigExporter(
        observations["summary"],
        observations["breakthrough"],
    )
    bulk_exporter.write_csv()
    bulk_exporter.print_bulk_config()


def convert_summary_to_bulk_config(args: Namespace) -> None:
    asyncio.run(_convert_summary_to_bulk_config(args.config))
