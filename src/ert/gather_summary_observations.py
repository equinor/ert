import asyncio
import datetime
import json
import ssl
import sys
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from dataclasses import dataclass, fields
from functools import partial
from pathlib import Path
from typing import Any, assert_never
from urllib.parse import quote

import anyio
import httpx
from resfo_utilities import SummaryKeyType

from ert.plugins import ErtRuntimePlugins


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


async def extract_summary_observations(
    ert_config: str, experiment: str
) -> dict[str, list[str]] | None:
    proc = None
    try:
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

        content = await anyio.Path(config_path).read_text()
        storage_config = json.loads(content)

        ssl_context = ssl.create_default_context()
        ssl_context.load_verify_locations(cafile=storage_config["cert"])
        ssl_context.check_hostname = False

        async with httpx.AsyncClient(
            base_url=storage_config["urls"][0],
            headers={"Token": storage_config["authtoken"]},
            verify=ssl_context,
        ) as client:
            experiments = json.loads((_ := await client.get("/experiments")).text)
            summary_observations = next(
                filter(lambda x: x["id"] == experiment, experiments)
            )["observations"]["summary"]
            response_key = {v: k for k, vs in summary_observations.items() for v in vs}
            result = defaultdict(list)
            for obs in json.loads(
                (_ := await client.get(f"/experiments/{experiment}/observations")).text
            ):
                if obs["name"] in response_key:
                    result[response_key[obs["name"]]].append(obs)
            return result
    finally:
        if proc is not None:
            proc.terminate()


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


def parser(prog: str) -> ArgumentParser:
    p = ArgumentParser(prog=prog, description=__doc__)
    p.add_argument("ert_config", type=str)
    p.add_argument("experiment", type=str)
    p.add_argument("--output-csv-file", default="summary_observations.csv", type=str)
    return p


def main(args: Namespace, _site_plugins: ErtRuntimePlugins | None = None) -> None:
    summary_observations = asyncio.run(
        extract_summary_observations(args.config, args.experiment)
    )
    if summary_observations is None:
        return

    convert_summary_observations(summary_observations, args.output_csv_file)


if __name__ == "__main__":
    args = parser(sys.argv[0]).parse_args(sys.argv[1:])
    main(args)
