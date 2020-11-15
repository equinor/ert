from ert_shared.ensemble_evaluator import config as evaluator_config
from ert_shared.ensemble_evaluator.entity.function_ensemble import (
    create_function_ensemble,
)
from ert_shared.ensemble_evaluator.evaluator import EnsembleEvaluator


def evaluate(inputs, fun, executor):
    ensemble = create_function_ensemble(fun=fun, inputs=inputs, executor=executor)

    config = evaluator_config.load_config()
    ee = EnsembleEvaluator(ensemble=ensemble, config=config)

    ee.run()
    ensemble.join()
    ee.stop()

    return ensemble.results()
