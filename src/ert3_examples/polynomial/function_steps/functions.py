def polynomial(coefficients: dict) -> tuple:
    x_range = tuple(range(10))
    result = tuple(
        coefficients["a"] * x ** 2 + coefficients["b"] * x + coefficients["c"]
        for x in x_range
    )
    return result
