import random
import time

from ert_shared.ensemble_evaluator import config as evaluator_config
from ert_shared.ensemble_evaluator.entity.function_ensemble import (
    create_function_ensemble,
)
from ert_shared.ensemble_evaluator.evaluator import EnsembleEvaluator


def polynomial(coefficients, x_range=(2,)):
    return {
        "polynomial_output": [
            coefficients["a"] * (x ** 2) + coefficients["b"] * x + coefficients["c"]
            for x in x_range
        ]
    }


def generate_coefficients(n):
    return [
        {
            "coefficients": {
                "a": i,
                "b": i,
                "c": i,
            }
        }
        for i in range(n)
    ]


def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)


def generate_fib_inputs(n):
    return [{"n": _n} for _n in range(n)]


def evaluate(inputs, fun):
    ensemble = create_function_ensemble(fun=fun, inputs=inputs)

    config = evaluator_config.load_config()
    ee = EnsembleEvaluator(ensemble=ensemble, config=config)

    ee.run()
    ensemble.join()
    ee.stop()

    return ensemble.results()


def test_polynomial():
    inputs = generate_coefficients(10)
    results = evaluate(inputs, fun=polynomial)
    expected_results = [
        {"polynomial_output": [0]},
        {"polynomial_output": [7]},
        {"polynomial_output": [14]},
        {"polynomial_output": [21]},
        {"polynomial_output": [28]},
        {"polynomial_output": [35]},
        {"polynomial_output": [42]},
        {"polynomial_output": [49]},
        {"polynomial_output": [56]},
        {"polynomial_output": [63]},
    ]
    for r, r_expected in zip(results, expected_results):
        assert r == r_expected


def test_fibonacci():
    inputs = generate_fib_inputs(10)
    results = evaluate(inputs, fun=fibonacci)
    expected_results = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    for r, r_expected in zip(results, expected_results):
        assert r == r_expected
