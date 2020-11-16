def dummy_evaluator(coefficients):
    data = []
    for (a, b, c) in coefficients:
        data.append(
            {
                "polynomial_output": [a * x ** 2 + b * x + c for x in range(10)],
            }
        )
    return data
