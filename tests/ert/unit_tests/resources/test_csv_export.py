import datetime as _datetime
import json

# Inlined observation helpers (originally in ert.storage.observation_helpers)
from typing import Any

import polars as pl

from ert.plugins.hook_implementations.workflows.csv_export import CSVExportJob
from ert.storage import open_storage
from tests.ert.performance_tests.test_obs_and_responses_performance import (
    create_experiment_args,
)


def _to_iso_date(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        try:
            return value.date().isoformat()
        except Exception:
            return value.isoformat()
    try:
        import pandas as pd

        if isinstance(value, pd.Timestamp):
            return value.date().isoformat()
    except Exception:
        pass
    try:
        ms = int(value)
        dt = _datetime.datetime.fromtimestamp(ms / 1000.0)
        return dt.date().isoformat()
    except Exception:
        return str(value)


def dataframe_to_declarations(
    group_name: str, df: pl.DataFrame
) -> list[dict[str, Any]]:
    decls: list[dict[str, Any]] = []
    for row in df.iter_rows(named=True):
        row = dict(row)
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
                }
            )
            continue
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
                }
            )
            continue
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


def dataframes_to_declarations(dfs: dict[str, pl.DataFrame]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, df in dfs.items():
        out.extend(dataframe_to_declarations(name, df))
    return out


def test_that_csv_export_matches_snapshot(monkeypatch, tmp_path, snapshot):
    monkeypatch.chdir(tmp_path)
    num_realizations = 3
    info = create_experiment_args(
        num_parameters=5,
        num_gen_data_keys=5,
        num_gen_data_report_steps=2,
        num_gen_data_index=2,
        num_gen_data_obs=1,
        num_summary_keys=3,
        num_summary_timesteps=3,
        num_summary_obs=10,
        num_realizations=num_realizations,
    )

    with open_storage(tmp_path / "storage", mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "response_configuration": [info.gen_data_config, info.summary_config],
                "parameter_configuration": info.gen_kw_configs,
                "observations": dataframes_to_declarations(
                    {
                        "gen_data": info.gen_data_observations,
                        "summary": info.summary_observations,
                    }
                ),
            }
        )
        ens = experiment.create_ensemble(
            ensemble_size=num_realizations, name="BobKaareJohnny"
        )

        for real in range(num_realizations):
            ens.save_response("summary", info.summary_responses.clone(), real)
            ens.save_response("gen_data", info.gen_data_responses.clone(), real)

        ens.save_parameters(dataset=info.genkw_data)
        output_file = "the_export.csv"
        ensemble_list_json = json.dumps([str(ens.id)])
        CSVExportJob().run(storage, [output_file, ensemble_list_json])

        df = pl.read_csv(output_file).with_columns(pl.col(pl.Float64).round(2))

        snapshot.assert_match(
            df.write_csv(include_header=True), "csv_export_result.csv"
        )
