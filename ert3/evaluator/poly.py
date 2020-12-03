def polynomial(coefficients, x_range=tuple(range(10))):
    return {
        "polynomial_output": [
            coefficients["a"] * (x ** 2) + coefficients["b"] * x + coefficients["c"]
            for x in x_range
        ]
    }
