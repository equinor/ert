from __future__ import annotations

from typing import Dict, List, Any

import datetime
import polars as pl


def _to_iso_date(value: Any) -> str:
    # Accept python datetime, polars datetime, or integer milliseconds
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        # datetime.date or datetime.datetime
        try:
            return value.date().isoformat()
        except Exception:
            return value.isoformat()
    # polars may return pandas.Timestamp
    try:
        import pandas as pd

        if isinstance(value, pd.Timestamp):
            return value.date().isoformat()
    except Exception:
        pass
    # integer milliseconds since epoch
    try:
        ms = int(value)
        dt = datetime.datetime.fromtimestamp(ms / 1000.0)
        return dt.date().isoformat()
    except Exception:
        return str(value)


def dataframe_to_declarations(group_name: str, df: pl.DataFrame) -> List[Dict[str, Any]]:
    """Convert a single observation DataFrame into a flat list of pydantic-style declarations.

    The output dicts follow the `Observation` pydantic models in `ert.config._observations`.
    """
    decls: List[Dict[str, Any]] = []
    for row in df.iter_rows(named=True):
        row = dict(row)

        # Summary observations have a `time` and `response_key`/`observation_key`.
        if "time" in row and "response_key" in row:
            date = _to_iso_date(row.get("time"))
            decls.append(
                {
                    "type": "summary_observation",
                    "name": row.get("observation_key") or group_name,
                    "key": row.get("response_key"),
                    "date": date,
                    "value": float(row.get("observations")),
                    "error": float(row.get("std")),
                    "east": None if row.get("east") is None else float(row.get("east")),
                    "north": None if row.get("north") is None else float(row.get("north")),
                    "radius": None if row.get("radius") is None else float(row.get("radius")),
                }
            )
            continue

        # General observations (gen_data) contain index/report_step and value/std
        if "index" in row or "report_step" in row:
            decls.append(
                {
                    "type": "general_observation",
                    "name": row.get("observation_key") or group_name,
                    "data": row.get("response_key"),
                    "value": float(row.get("observations")),
                    "error": float(row.get("std")),
                    "restart": int(row.get("report_step", 0) or 0),
                    "index": int(row.get("index", 0) or 0),
                    "east": None if row.get("east") is None else float(row.get("east")),
                    "north": None if row.get("north") is None else float(row.get("north")),
                    "radius": None if row.get("radius") is None else float(row.get("radius")),
                }
            )
            continue

        # RFT observations: response_key is usually "WELL:DATE:PROPERTY"
        if "tvd" in row or "well" in row or ("response_key" in row and ":" in str(row.get("response_key", ""))):
            resp = str(row.get("response_key", ""))
            try:
                well, date_str, prop = resp.split(":", 2)
            except Exception:
                well = row.get("well") or group_name
                date_str = _to_iso_date(row.get("time") or row.get("date"))
                prop = row.get("property") or "PRESSURE"
            decls.append(
                {
                    "type": "rft_observation",
                    "name": row.get("observation_key") or group_name,
                    "well": well,
                    "date": date_str,
                    "property": prop,
                    "value": float(row.get("observations")),
                    "error": float(row.get("std")),
                    "north": None if row.get("north") is None else float(row.get("north")),
                    "east": None if row.get("east") is None else float(row.get("east")),
                    "tvd": None if row.get("tvd") is None else float(row.get("tvd")),
                }
            )
            continue

        # Fallback: place the row as a general observation with minimal fields
        decls.append(
            {
                "type": "general_observation",
                "name": group_name,
                "data": row.get("response_key") or "",
                "value": float(row.get("observations", 0.0) or 0.0),
                "error": float(row.get("std", 1.0) or 1.0),
                "restart": int(row.get("report_step", 0) or 0),
                "index": int(row.get("index", 0) or 0),
            }
        )

    return decls


def dataframes_to_declarations(dfs: Dict[str, pl.DataFrame]) -> List[Dict[str, Any]]:
    """Convert a mapping of observation group -> DataFrame into a flat list of declarations."""
    out: List[Dict[str, Any]] = []
    for name, df in dfs.items():
        out.extend(dataframe_to_declarations(name, df))
    return out
