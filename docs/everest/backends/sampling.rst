.. _cha_sampling_backends:


Sampling backends
=================

By default Everest generates perturbations to estimate gradients using sampling
code from SciPy. It provides sampling from common distributions such as
Gaussian, Uniform and Bernoulli, and some additional methods such as Sobol and
Latin hypercube sampling.

The sampling method and options are specified in the `sampler` subsection of the
`controls` configuration settings. The sampling method is selected using the
`method` keyword. If the `method` keyword is missing, the backend will select a
default method, which samples from a normal distribution.

The sampling methods in the SciPy backend support several options that can be
passed using the `options` keyword. Please consult the online SciPy manual for
details for the options to those sampler that can be passed.

Everest supports the possibility to install additional samplers via a plugin
mechanism. The correct plugin to use will generally be inferred from the method
name. If several plugins have a method named `A`, pick a specific backend `B`
by putting `B/A` in the `method` field.

**Example**

.. code-block:: yaml

    optimization:
        sampler:
            method: sobol
            options:
                scramble: False
