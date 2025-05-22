from __future__ import annotations

from pydantic import Field

from ert.run_models import EnsembleExperiment

SINGLE_TEST_RUN_GROUP = "Forward model evaluation"


class SingleTestRun(EnsembleExperiment):
    """
    Single test is equivalent to EnsembleExperiment, in that it
    samples the prior and evaluates it.<br>There are two key differences:<br>
    1) Single test run always runs locally using the <b>local queue</b><br>
    2) Only a <b>single realization</b> (realization-0) is run<br>
    """

    active_realizations: list[bool] = Field(default_factory=lambda: [True])
    minimum_required_realizations: int = 1

    @classmethod
    def name(cls) -> str:
        return "Single realization test-run"

    @classmethod
    def description(cls) -> str:
        return "Sample parameters → evaluate single realization"

    @classmethod
    def group(cls) -> str | None:
        return SINGLE_TEST_RUN_GROUP
