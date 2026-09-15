import json

import polars as pl

from ert.plugins.hook_implementations.workflows.csv_export import CSVExportJob
from ert.storage import open_storage
from ert.storage.local_ensemble import LocalEnsemble
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
            experiment_config={
                "response_configuration": [info.gen_data_config, info.summary_config],
                "parameter_configuration": info.gen_kw_configs,
                "observations": info.gen_data_observations + info.summary_observations,
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


def test_that_csv_export_stops_processing_ensembles_once_cancelled(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    num_realizations = 2
    info = create_experiment_args(
        num_parameters=1,
        num_gen_data_keys=1,
        num_gen_data_report_steps=1,
        num_gen_data_index=1,
        num_gen_data_obs=1,
        num_summary_keys=1,
        num_summary_timesteps=1,
        num_summary_obs=1,
        num_realizations=num_realizations,
    )

    job = CSVExportJob()
    processed_ensembles = []
    original_has_data = LocalEnsemble.has_data

    def tracking_has_data(self):
        processed_ensembles.append(self.name)
        job.cancel()
        return original_has_data(self)

    monkeypatch.setattr(LocalEnsemble, "has_data", tracking_has_data)

    with open_storage(tmp_path / "storage", mode="w") as storage:
        experiment = storage.create_experiment(
            experiment_config={
                "response_configuration": [info.gen_data_config, info.summary_config],
                "parameter_configuration": info.gen_kw_configs,
                "observations": info.gen_data_observations + info.summary_observations,
            }
        )
        ensembles = [
            experiment.create_ensemble(ensemble_size=num_realizations, name=name)
            for name in ("first", "second")
        ]
        for ens in ensembles:
            for real in range(num_realizations):
                ens.save_response("summary", info.summary_responses.clone(), real)
                ens.save_response("gen_data", info.gen_data_responses.clone(), real)
            ens.save_parameters(dataset=info.genkw_data)

        ensemble_list_json = json.dumps([str(ens.id) for ens in ensembles])
        job.run(storage, ["the_export.csv", ensemble_list_json])

    # Cancellation is noticed at the top of the next iteration, so the
    # ensemble being processed when cancel() was called still completes,
    # but the one after it is skipped entirely.
    assert processed_ensembles == ["first"]

    df = pl.read_csv("the_export.csv")
    assert set(df["Ensemble"]) == {"first"}
