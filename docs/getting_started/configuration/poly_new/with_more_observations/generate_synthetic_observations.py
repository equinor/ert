import numpy as np

rng = np.random.default_rng(12345)


def p(x):
    return 0.5 * x**2 + x + 3


data_points = [
    (p(x) + rng.normal(loc=0, scale=0.10 * p(x)), 0.10 * p(x)) for x in range(50)
]

# Format the data points with each pair on a separate line
formatted_data = "\n".join(f"{value[0]} {value[1]:.1f}" for value in data_points)

print(formatted_data)
