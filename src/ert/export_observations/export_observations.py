import asyncio
import json
import ssl
from argparse import Namespace
from collections import defaultdict
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from ssl import SSLContext
from typing import Any, Literal
from urllib.parse import quote

import anyio
import httpx
from httpx import AsyncClient

from ert.cli.main import ErtCliError
from ert.export_observations.bulk_config_exporter import BulkConfigExporter


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


async def collect_all_observations(client: AsyncClient, args: Namespace) -> ObsDict:
    observations: ObsDict = {}
    all_experiments = await fetch_experiments(client)
    experiment_id = get_experiment_id(all_experiments, args.experiment_id)
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
        async with connect(
            storage_config["urls"], storage_config["authtoken"], ssl_certificate
        ) as client:
            observations = await collect_all_observations(client, args)
    finally:
        if proc is not None and proc.returncode is None:
            proc.terminate()
            await proc.wait()

    bulk_exporter = BulkConfigExporter(
        observations["summary"],
        observations["breakthrough"],
        args.output_csv_file,
    )
    bulk_exporter.write_csv()
    bulk_exporter.print_bulk_config()


def main(args: Namespace, _site_plugins: Any | None = None) -> None:
    asyncio.run(_async_main(args))
