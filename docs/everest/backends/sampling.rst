.. _cha_sampling_backends:


Sampling backends
=================

Many optimization algorithms require the gradient of the objective function.
These are estimated using evaluations of perturbed controls, which requires
(pseudo-)stochastic sampling. By default these perturbations are generated using
sampling code from the `SciPy <https://www.scipy.org>`_ package. SciPy provides
sampling from common distributions such as Gaussian, Uniform and Bernoulli, and
some additional methods such as Sobol and Latin hypercube sampling.

The sampling method and options are specified in the ``sampler`` subsection of
the `controls` configuration settings. The sampling method is selected using the
``method`` keyword. If the ``method`` keyword is missing, a normal distribution
is used.

The sampling methods in the SciPy backend support several options that can be
passed using the ``options`` keyword. Consult the online SciPy manual on
`Statistical functions <https://docs.scipy.org/doc/scipy/reference/stats.html>`_
for details. To find the algorithms and options that are supported in EVEREST,
consult the relevant section of the ``ropt`` manual: `SciPy Sampler Plugin
<https://tno-ropt.github.io/ropt/dev/reference/scipy_sampler_plugin/>`_.

**Example**

.. code-block:: yaml

    optimization:
        sampler:
            method: sobol
            options:
                scramble: False
