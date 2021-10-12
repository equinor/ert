from typing import Dict, Tuple


def polynomial(coefficients: Dict[str, float]) -> Dict[str, Tuple]:
    x_range = tuple(range(10))
    result = tuple(
        coefficients["a"] * x ** 2 + coefficients["b"] * x + coefficients["c"]
        for x in x_range
    )
    return {"polynomial_output": result}
