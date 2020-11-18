from collections.abc import Iterable


def evaluate(coefficients):
    data = []
    if not isinstance(coefficients, Iterable):
        raise ValueError(f"Input must be an iterable, was {coefficients}")

    for realization in coefficients:
        try:
            coeffs = realization["coefficients"]
        except KeyError as err:
            raise ValueError(
                f"Each entry in the input must be a dict with key <coefficients>, had {realization.keys()}"
            ) from err
        except TypeError as err:
            raise ValueError(
                f"Each entry in the input must be a dict, was {realization}, with type {type(realization)}"
            ) from err

        if not isinstance(coeffs, dict):
            raise ValueError(
                f"Each coefficients entry in the input must be a dict, was {coeffs}, type {type(coeffs)})"
            )
        if set(coeffs.keys()) != set(("a", "b", "c")):
            raise ValueError(
                f"Each coefficients entry in the input must contain only the keys <a>, <b> and <c>, had {coeffs.keys()}"
            )

        try:
            data.append(
                {
                    "polynomial_output": [
                        coeffs["a"] * x ** 2 + coeffs["b"] * x + coeffs["c"]
                        for x in range(10)
                    ],
                }
            )
        except TypeError:
            raise ValueError(
                f"The content of the coefficients for each <a>, <b> and <c> must be a number, was {coeffs.items()}"
            )

    return data
