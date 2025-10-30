.. _cha_optimization_backends:

Optimization backends
=====================

Everest offers various optimization backends that allow the user a wide
selection of low-level optimization algorithms to perform ensemble optimization.
Most of the options that can be set in the ``optimization`` section will be
implemented by all algorithms. These options are documented in the section
:ref:`cha_config_reference`. In addition to these standard option it is possible
to pass generic options via the ``options`` keyword. To find the generic options
that are supported by a backend, consult their documenation.


Out of the box, Everest supports Dakota and SciPy backends.

The Dakota backend
------------------

The Dakota backend is based on the `Dakota <https://dakota.sandia.gov/>`_
optimization package. Information on two commonly used algorithms
(`optpp_q_newton` and `conmin_mfd`) can be found in
:ref:`cha_optimization_algorithms`.

.. important::

	For more information regarding specific optimizer settings please refer to
	the official `Dakota manual <https://snl-dakota.github.io>`_. To find the
	algorithms and options that are supported in Everest, consult the manual of
	the corresponding ``ropt`` plugin: `ropt-dakota
	<https://tno-ropt.github.io/ropt-dakota>`_.


**Example**

.. code-block:: yaml

    optimization:
        algorithm: optpp_q_newton
        convergence_tolerance: 0.001
        constraint_tolerance: 0.001
        perturbation_num: 7
        speculative: True
        options:
            - max_repetitions = 300
            - retry_if_fail
            - classical_search 1

.. note::

    The ``constraint_tolerance`` option used in this example is specific for
    Dakota, which uses it to determine if output constraints are violated.

The SciPy backend
-----------------

The SciPy backend is based on the optimization algorithms implemented in the
`SciPy <https://www.scipy.org>`_ package.

.. important::

	For more information regarding specific optimizer settings please refer to
	the `scipy.optimize
	<https://docs.scipy.org/doc/scipy/tutorial/optimize.html>`_ manual. To find
	the algorithms and options that are supported in Everest, consult the
	``ropt`` manual: `https://scipy.org/
	<https://tno-ropt.github.io/ropt/dev/reference/scipy_optimizer_plugin/>`_.

**Example**

.. code-block:: yaml

    optimization:
        backend: scipy
        algorithm: SLSQP
        convergence_tolerance: 0.001
        perturbation_num: 7
        speculative: True
        backend_options:
            ftol: 1e-5
            disp: True
