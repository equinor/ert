from datetime import datetime

import polars as pl

from ert.config import BreakthroughConfig, SummaryConfig
from ert.storage.local_storage import open_storage


def test_that_derive_from_storage_frames_are_stackable_regardless_of_breakthrough_time(
    tmp_path,
):
    with open_storage(tmp_path, mode="w") as storage:
        response_key = "WWCT:OP1"
        time = datetime(2000, 3, 2, 13, 0)  # ruff: ignore[call-datetime-without-tzinfo]

        breakthrough_config = BreakthroughConfig(
            keys=[f"BREAKTHROUGH:{response_key}"],
            summary_keys=[response_key],
            thresholds=[0.2],
            observed_dates=[time],
        )

        summary_config = SummaryConfig(
            keys=[response_key],
            input_files=["not_relevant"],
        )

        experiment = storage.create_experiment(
            experiment_config={
                "response_configuration": [
                    summary_config.model_dump(mode="json"),
                    breakthrough_config.model_dump(mode="json"),
                ],
            }
        )

        ensemble = storage.create_ensemble(
            experiment, ensemble_size=2, iteration=0, name="prior"
        )

        def create_summary_response_dataframe(
            response_key: str, realization: int, value_modifier
        ) -> pl.DataFrame:
            return pl.DataFrame(
                {
                    "realization": [realization] * 5,
                    "response_key": [response_key] * 5,
                    "time": [datetime(2000, month, 1, 1, 0) for month in range(1, 6)],  # ruff: ignore[call-datetime-without-tzinfo]
                    "values": [n / value_modifier for n in range(5)],
                }
            )

        value_over_threshold = create_summary_response_dataframe(response_key, 0, 10)
        value_under_threshold = create_summary_response_dataframe(response_key, 1, 100)

        ensemble.save_response("summary", value_over_threshold, 0)
        ensemble.save_response("summary", value_under_threshold, 1)

        breakthrough_response0 = breakthrough_config.derive_from_storage(0, 0, ensemble)
        breakthrough_response1 = breakthrough_config.derive_from_storage(0, 1, ensemble)

        ensemble.save_response("breakthrough", breakthrough_response0, 0)
        ensemble.save_response("breakthrough", breakthrough_response1, 1)

        responses = ensemble.load_responses("breakthrough", (0, 1))
        assert len(responses) == 2
