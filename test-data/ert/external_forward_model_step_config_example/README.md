## Polynomial curve fitting - A minimal model updating case
A display of a truly minimal model updating case. It is done with a second
degree polynomial as the _true reality_. The model is `ax^2 + bx + c` where
`a`, `b` and `c` are the parameters.

### Observed data
The observed data was generated with the following _Python_ code:

```python
import numpy as np

rng = np.random.default_rng(12345)


def p(x):
    return 0.5 * x ** 2 + x + 3


[(p(x) + rng.normal(loc=0, scale=0.2 * p(x)), 0.2 * p(x)) for x in [0, 2, 4, 6, 8]]
```

This gives us observations (both a value and an uncertainty) for even `x` less
than 10. These values appear in `poly_obs_data.txt`. Finally, these values are
represented as an observation in `observations`, where they are given an identifier.
And we specify that the values that should be used from the `forward_model` are only
the even ones (the `forward model` spits out the image of the polynomial on the
range `[0, 9]`).

### Parameters
As mentioned above `a`, `b` and `c` form the parameters of the model `ax^2 + bx + c`.
They are all specified to be uniformly distributed over ranges in
`coeff_priors` and are sampled by `GEN_KW`, they are provided to the forward model
through `parameters.json`.

### Forward model
After the parameters are dumped to the runpath, _forward model_'s are launched
for each of the realizations. The forward model consists of a single script
described in `poly_eval.py`, that loads the dumped parameters and outputs the
values of the polynomial (given the parameters) for integer `x in [0, 10]` to the
file `poly.out`.

The very minimal job description file `POLY_EVAL` just points to the script.

### Loading data
The configuration specifies a `GEN_DATA` that expects there to be a result file
`poly.out`. In other words it expects to load data from `poly.out`, the exact
file that the forward model produces.

### Model update
Then the loaded data is compared to the observed data and the parameters are
updated accordingly.
