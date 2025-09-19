import json

import polars as pl

from ert.plugins.hook_implementations.workflows.csv_export import CSVExportJob
from ert.storage import open_storage
from tests.ert.performance_tests.test_obs_and_responses_performance import (
    create_experiment_args,
)


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
            responses=[info.gen_data_config, info.summary_config],
            parameters=info.gen_kw_configs,
            observations={
                "gen_data": info.gen_data_observations,
                "summary": info.summary_observations,
            },
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
