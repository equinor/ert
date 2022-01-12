## Polynomial curve fitting - A minimal model updating case
A display of a truly minimal model updating case. It is done with a second
degree polynomial as the _true reality_. The model is `ax^2 + bx + c` where
`a`, `b` and `c` are the parameters.

### Observed data
The observed data was generated with the following _Python_ code:
```
def p(x):
     return 0.5*x**2 + x + 3

[(p(x)+random.gauss(0, 0.25*x**2+0.1), 0.25*x**2+0.1) for x in [0, 2, 4, 6, 8]]
```

This gives us observations (both a value and an uncertainty) for even `x` less
than 10. These values appear in `poly_obs_data.txt`. Finally, these values are
represented as an observation in `observations`. Here we give the data a name.
And we specify that the values that should be used from the `forward_model` is only
the even ones (the `forward model` spits out the image of the polynomial on the
range `[0, 9]`). It also specifies that the time step to consider is `0`
(`RESTART`). We do not really have a time concept in this setup and hence we
only use `0` as a dummy value. We could of course have considered the values
feed to the polynomial as time; but that is left as an exercise for the reader.

### Parameters
As mentioned above `a`, `b` and `c` forms the parameters of the model `ax^2 + bx + c`.
They are all specified to be uniformly distributed over ranges in
`coeff_priors` and are sampled by `GEN_KW` and dumped to the forward model
following the `json`-template in `coeff.tmpl`.

### Forward model
After the parameters are dumped to the runpath, _forward model_'s are launched
for each of the realizations. The forward model consists of a single script
described in `poly_eval.py`, that loads the dumped parameters and outputs the
values of the polynomial (given the parameters) for integer `x in [0, 10]` to the
file `poly_0.out`.

The very minimal job description file `POLY_EVAL` just points to the script.

### Loading data
The configuration specifies a `GEN_DATA` that expects there to be a result file
`poly_%d.out` for report step `0`. In other words it expects to load data from
`poly_0.out`, the exact file that the forward model produces.

### Model update
Then the loaded data is compared to the observed data and the parameters are
updated accordingly.

### Time
Although we don't really have a clear concept of time in this case _ERT_
expects us to have so. Since we have specified that all of our data is for time
step `0`, we had to create an artificial time map `time_map` that specifies
that the very first and only report step corresponds to `1/10/2006`. This could
be any date, so we put it to the date of the first commit of the
`ensembles/ert` repository on _Github_.
