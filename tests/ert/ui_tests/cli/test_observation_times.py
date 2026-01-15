"""
Tests behavior of matching response times to observation times
"""

from contextlib import redirect_stderr
from datetime import date, datetime, timedelta
from io import StringIO
from textwrap import dedent

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import assume, given, settings
from resfo_utilities.testing import summaries

from ert.cli.main import ErtCliError
from ert.mode_definitions import ES_MDA_MODE
from tests.ert.unit_tests.config.observations_generator import (
    as_obs_config_content,
    summary_observations,
)

from .run_cli import run_cli

start = datetime(1969, 1, 1)
observation_times = st.dates(
    min_value=date.fromordinal((start + timedelta(hours=1)).toordinal()),
    max_value=date(2024, 1, 1),
).map(lambda x: datetime.fromordinal(x.toordinal()))


@pytest.mark.filterwarnings(
    "ignore:.*overflow encountered in multiply.*:RuntimeWarning"
)
@pytest.mark.usefixtures("use_site_configurations_with_no_queue_options")
@settings(max_examples=3)
@given(
    responses_observation=observation_times.flatmap(
        lambda observation_time: st.fixed_dictionaries(
            {
                "responses": st.lists(
                    summaries(
                        start_date=st.just(start),
                        time_deltas=st.just(
                            [((observation_time - start).total_seconds()) / 3600]
                        ),
                        summary_keys=st.just(["FOPR"]),
                        use_days=st.just(False),
                    ),
                    min_size=2,
                    max_size=2,
                ),
                "observation": summary_observations(
                    summary_keys=st.just("FOPR"),
                    std_cutoff=10.0,
                    names=st.just("FOPR_OBSERVATION"),
                    datetimes=st.just(observation_time),
                ),
            }
        )
    ),
    std_cutoff=st.floats(min_value=1e-6, max_value=1.0),
    enkf_alpha=st.floats(min_value=3.0, max_value=10.0),
    epsilon=st.sampled_from([0.0, 1.1, 2.0, -2.0]),
)
def test_that_small_time_mismatches_in_summaries_are_ignored(
    responses_observation, tmp_path_factory, std_cutoff, enkf_alpha, epsilon
):
    responses = responses_observation["responses"]
    observation = responses_observation["observation"]
    tmp_path = tmp_path_factory.mktemp("summary")
    (tmp_path / "config.ert").write_text(
        dedent(
            f"""
            NUM_REALIZATIONS 2
            QUEUE_SYSTEM LOCAL
            QUEUE_OPTION LOCAL MAX_RUNNING 2
            ECLBASE CASE
            SUMMARY FOPR
            MAX_SUBMIT 1
            GEN_KW KW_NAME prior.txt
            OBS_CONFIG observations.txt
            STD_CUTOFF {std_cutoff}
            ENKF_ALPHA {enkf_alpha}
            """
        )
    )

    # Add some inprecision to the reported time
    for r in responses:
        r[1].steps[-1].ministeps[-1].params[0] += epsilon

    (tmp_path / "prior.txt").write_text("KW_NAME NORMAL 0 1")
    response_values = np.array(
        [r[1].steps[-1].ministeps[-1].params[-1] for r in responses]
    )
    std_dev = response_values.std(ddof=0)
    assume(np.isfinite(std_dev))
    assume(std_dev > std_cutoff)
    observation.value = float(response_values.mean())
    for i in range(2):
        for j in range(4):
            summary = responses[i]
            smspec, unsmry = summary
            (tmp_path / f"simulations/realization-{i}/iter-{j}").mkdir(parents=True)
            smspec.to_file(
                tmp_path / f"simulations/realization-{i}/iter-{j}/CASE.SMSPEC"
            )
            unsmry.to_file(
                tmp_path / f"simulations/realization-{i}/iter-{j}/CASE.UNSMRY"
            )
    (tmp_path / "observations.txt").write_text(as_obs_config_content(observation))

    if abs(epsilon) < 1 / 3600:  # less than one second
        stderr = StringIO()
        with redirect_stderr(stderr):
            run_cli(
                ES_MDA_MODE,
                str(tmp_path / "config.ert"),
                "--weights=0,1",
            )
        assert "Experiment completed" in stderr.getvalue()
    else:
        with pytest.raises(ErtCliError, match="No active observations"):
            run_cli(
                ES_MDA_MODE,
                "--disable-monitoring",
                str(tmp_path / "config.ert"),
                "--weights=0,1",
            )
