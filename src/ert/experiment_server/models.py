from uuid import UUID
from typing import Union, List

from pydantic import BaseModel, Field

from ert.config import ErtConfig
from ert.gui.simulation.ensemble_experiment_panel import (
    Arguments as EnsembleExperimentArguments,
)
from ert.gui.simulation.ensemble_smoother_panel import (
    Arguments as EnsembleSmootherArguments,
)
from ert.gui.simulation.evaluate_ensemble_panel import (
    Arguments as EvaluateEnsembleArguments,
)
from ert.gui.simulation.iterated_ensemble_smoother_panel import (
    Arguments as IteratedEnsembleSmootherArguments,
)
from ert.gui.simulation.manual_update_panel import Arguments as ManualUpdateArguments
from ert.gui.simulation.multiple_data_assimilation_panel import (
    Arguments as MultipleDataAssimilationArguments,
)
from ert.gui.simulation.single_test_run_panel import Arguments as SingleTestRunArguments

class Experiment(BaseModel):
    args: Union[
        EnsembleExperimentArguments,
        EnsembleSmootherArguments,
        EvaluateEnsembleArguments,
        IteratedEnsembleSmootherArguments,
        ManualUpdateArguments,
        MultipleDataAssimilationArguments,
        SingleTestRunArguments,
    ] = Field(..., discriminator="mode")
    ert_config: ErtConfig

class ExperimentOut(BaseModel):
   id: UUID
   type: str
