from ert_shared.ensemble_evaluator import config as evaluator_config
from ert_shared.ensemble_evaluator.entity.function_ensemble import (
    create_function_ensemble,
)
from ert_shared.ensemble_evaluator.evaluator import EnsembleEvaluator


def _polynomial(coefficients, x_range=tuple(range(10))):
    return {
        "polynomial_output": [
            coefficients["a"] * (x ** 2) + coefficients["b"] * x + coefficients["c"]
            for x in x_range
        ]
    }


def evaluate(inputs, fun=None):
    if fun == None:
        fun = _polynomial

    ensemble = create_function_ensemble(fun=fun, inputs=inputs, executor="local")

    config = evaluator_config.load_config()
    ee = EnsembleEvaluator(ensemble=ensemble, config=config)

    ee.run()
    ensemble.join()
    ee.stop()

    return ensemble.results()
