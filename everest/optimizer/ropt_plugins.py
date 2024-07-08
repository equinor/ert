import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Optional, cast

from ert import WorkflowRunner
from ert.config import ErtConfig
from ert.storage import open_storage
from pydantic import BaseModel, ConfigDict
from ropt.config.plan import EvaluatorStepConfig
from ropt.exceptions import ConfigError
from ropt.optimization import BasicStep, Plan, PlanContext
from ropt.plugins.optimization_steps.base import (
    OptimizationSteps,
    OptimizationStepsPlugin,
)
from ropt.plugins.optimization_steps.evaluator import DefaultEvaluatorStep
from ropt.results import FunctionResults, convert_to_maximize

from everest.config import EverestConfig
from everest.simulator.everest2res import everest2res


class WorkflowJobConfig(BaseModel):
    job: str
    output: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class EnsembleEvaluationStepConfig(EvaluatorStepConfig):
    results: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class WorkflowJob(BasicStep):
    def __init__(
        self, config: Dict[str, Any], plan: Plan, ever_config: EverestConfig
    ) -> None:
        self._config = WorkflowJobConfig.model_validate(config)
        self._plan = plan
        self._ert_dict = everest2res(
            ever_config, site_config=ErtConfig.read_site_config()
        )

    def run(self) -> bool:
        job = self._config.job
        if self._config.output is not None:
            job = f"{job} -o {self._config.output}"

        # Make a temporary workflow file:
        with NamedTemporaryFile(
            "w", encoding="utf-8", suffix=".workflow", delete=False
        ) as fp:
            file_name = Path(fp.name)
            workflow_name = file_name.stem
            fp.write(job)

        # Run the user job:
        try:
            workflows = self._ert_dict.get("LOAD_WORKFLOW", [])
            self._ert_dict["LOAD_WORKFLOW"] = workflows + [(file_name, workflow_name)]
            ert_config = ErtConfig.with_plugins().from_dict(config_dict=self._ert_dict)
            with open_storage(ert_config.ens_path, "w") as storage:
                workflow = ert_config.workflows[workflow_name]
                runner = WorkflowRunner(workflow, storage, None, ert_config)
                runner.run_blocking()
                if not all(v["completed"] for v in runner.workflowReport().values()):
                    raise RuntimeError("workflow job failed")
        finally:
            if workflows:
                self._ert_dict["LOAD_WORKFLOW"] = workflows
            file_name.unlink(missing_ok=True)

        # Read the script result and try to use it to modify the configuration:
        if self._config.output is not None and Path(self._config.output).exists():
            with open(self._config.output, "r", encoding="utf-8") as fp:
                job_results = json.load(fp)
            updates = {}

            # The workflow job may have stored new realization weights:
            realization_weights = job_results.get("realization_weights")
            if realization_weights is not None:
                updates["realizations"] = {"weights": realization_weights}

            # The workflow job may have stored new initial values:
            initial_variables = job_results.get("initial_variables")
            if initial_variables is not None:
                updates["variables"] = {"initial_values": initial_variables}

            # The workflow job may have stored updates for the optimizer options:
            optimizer_options = job_results.get("optimizer_options")
            if optimizer_options is not None:
                updates["optimizer"] = {"options": optimizer_options}

            # If there are any updates, inform the  plan about it, so that
            # future steps, like optimizations use the updated configurations:
            if updates:
                self._plan.update_enopt_config(updates)

        return False


class EnsembleEvaluationStep(DefaultEvaluatorStep):
    def __init__(
        self, config: Dict[str, Any], context: PlanContext, plan: Plan
    ) -> None:
        super().__init__(config, context, plan)

        self._config = EnsembleEvaluationStepConfig.model_validate(config)
        self._plan = plan

    # This runs after the ensemble evaluation, storing the results:
    def process(self, results: FunctionResults) -> None:
        if self._config.results is not None:
            # ropt is a minimizer, so we fix the results for Everest:
            results = cast(FunctionResults, convert_to_maximize(results))

            # Get some results to store:
            enopt_config = self._plan.enopt_config
            assert enopt_config is not None
            assert enopt_config.objective_functions.names is not None

            # Pick out the realization weights and the objectives:
            evaluation_results = {
                "realization_weights": enopt_config.realizations.weights.tolist(),
                "objectives": {
                    objective: results.evaluations.objectives[:, obj_idx].tolist()
                    for obj_idx, objective in enumerate(
                        enopt_config.objective_functions.names
                    )
                },
            }

            # See if there is a restart number to report:
            restart = results.metadata.get("restart")
            if restart is not None:
                evaluation_results["restart"] = restart

            # Dump the evaluation results to a JSON file:
            with open(self._config.results, "w", encoding="utf-8") as fp:
                json.dump(evaluation_results, fp)


class EverestPlanSteps(OptimizationSteps):
    def __init__(
        self, context: PlanContext, plan: Plan, ever_config: EverestConfig
    ) -> None:
        self._ever_config = ever_config
        self._context = context
        self._plan = plan

    def get_step(self, config: Dict[str, Any]) -> Any:
        # There should be a single entry that denotes the step type:
        keys = iter(config.keys())
        step_type = next(keys, None)
        if step_type is None:
            raise ConfigError("Not an optimization step")
        if next(keys, None) is not None:
            msg = f"Step type is ambiguous: {keys}"
            raise ConfigError(msg)

        # Handle the different types supported by this backend:
        if step_type == "workflow_job":
            return WorkflowJob(config["workflow_job"], self._plan, self._ever_config)

        if step_type == "ensemble_evaluation":
            return EnsembleEvaluationStep(
                config["ensemble_evaluation"], self._context, self._plan
            )

        msg = f"Step type not supported: {step_type}"
        raise ConfigError(msg)


class EverestPlanStepPlugin(OptimizationStepsPlugin):
    def __init__(self, ever_config: EverestConfig):
        self._ever_config = ever_config

    def create(self, context: PlanContext, plan: Plan) -> EverestPlanSteps:
        return EverestPlanSteps(context, plan, self._ever_config)

    def is_supported(self, method: str) -> bool:
        return method.lower() in {"workflow_job", "ensemble_evaluation"}
