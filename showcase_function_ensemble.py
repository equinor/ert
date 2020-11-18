import random
import time

from ert_shared.ensemble_evaluator import config as evaluator_config
from ert_shared.ensemble_evaluator.entity.function_ensemble import (
    create_function_ensemble,
)
from ert_shared.ensemble_evaluator.evaluator import EnsembleEvaluator


def polynomial(coefficients, x_range=tuple(range(11))):
    return {
        "polynomial_output": [
            coefficients["a"] * (x ** 2) + coefficients["b"] * x + coefficients["c"]
            for x in x_range
        ]
    }


def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)


def generate_fib_inputs():
    return [{"n": n} for n in range(40)]


def generate_coefficients():
    return [
        {
            "coefficients": {
                "a": random.gauss(0, 1),
                "b": random.gauss(0, 1),
                "c": random.gauss(0, 1),
            }
        }
        for _ in range(50)
    ]


def evaluate(inputs, fun):
    ensemble = create_function_ensemble(fun=fun, inputs=inputs, executor="local")

    config = evaluator_config.load_config()
    ee = EnsembleEvaluator(ensemble=ensemble, config=config)

    ee.run()
    ensemble.join()
    ee.stop()

    return ensemble.results()


def main():
    inputs = generate_coefficients()
    results = evaluate(inputs, fun=polynomial)
    print(results)

    inputs = generate_fib_inputs()
    results = evaluate(inputs, fun=fibonacci)
    print(results)


if __name__ == "__main__":
    main()
