********
Backends
********

.. toctree::
    :hidden:

    optimization
    sampling

Ensemble-based optimization is handled internally by the `ropt
<https://tno-ropt.github.io/ropt/>`_ library for robust optimization, which in
turn uses a plugin mechanism to provide multiple low-level optimization
backends. The different backends employ various well-tested optimization
packages, providing the user with a wide choice of reliable algorithms, see
section :ref:`cha_optimization_backends`.

Similarly, sampling backends are used to generate perturbations for gradient
estimation during optimization. These are described in section:
:ref:`cha_sampling_backends`.

Optimization algorithms are selected by name, by setting the ``algorithm`` field
in the ``optimization`` section of the Everest configuration file. Everest will
find the correct backend from the name of the algorithm. In the unlikely case
that multiple backends are installed that provide the same algorithm, it is
possible to specify the backend by name, by setting the ``algorithm`` field to a
value of the form ``backend-name/algorithm-name``, i.e. by pre-fixing the
algorithm name with the backend name, separated by a slash. Sampling methods are
selected in similar fashion, using the ``method`` field in the ``sampler``
sections of the Everest configuration file.
