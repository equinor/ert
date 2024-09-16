.. _cha_sampler_backends:

********
Samplers
********

By default Everest generates perturbations to estimate gradients using sampling
code based on SciPy. It provides sampling from common distributions such as
Gaussian, Uniform and Bernoulli, and some additional methods such as Sobol and
Latin hypercube sampling.

The sampling method and options are specified in the `sampler` subsection of the
`controls` configuration settings. The sampling method is selected using the
`method` keyword. If the `method` keyword is missing, the backend will select a
default method, which samples from a normal distribution.

The sampling methods in the SciPy backend support several options that can be
passed using the `backend_options` keyword. Please consult the online SciPy
manual for details for the options to those sampler that can be passed.

The sampler methods use a random number generator to generate stochastic
samples. The seed for that generator can be set using the `seed` keyword. If the
`seed` keyword is not provided, the more general Everest seed, as configured by
the `random_seed` keyword is used instead.

Everest offers the possibility to specify another backend for sampling with the
optional `backend` keyword in the `sampler`` subsection of the 'controls'
configuration settings. Everest will check if the requested backend is
installed, and if the method specified by the `method` keyword is supported. By
default, if the `backend` keyword is missing, Everest will select the `scipy`
backend. Additional backends may be installed in your local installation,
consult your systems manager for options.

**Example**

.. code-block:: yaml

    optimization:
        sampler:
            method: sobol
            seed: 123

Another example using the SciPy backend, with some options added, and the seed
taken from the global Everest seed:

.. code-block:: yaml

    optimization:
        sampler:
            method: sobol
            backend_options:
                scramble: False
