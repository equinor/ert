.. _cha_optimization_backends:

Optimization backends
=====================

Everest offers the possibility to specify the backend used for optimization with
the `backend` keyword. Everest will check if the requested backend is installed
and if the algorithm specified by the `algorithm` keyword is supported. The
general optimization options, like `convergence_tolerance` are handled
appropiately by the backend, and backend specific options can be passed using
the `options` or `backend_options` keywords.

Out of the box, Everest supports Dakota and SciPy backends, provided their
corresponding prerequisites are installed, i.e., provided Dakota and/or Scipy are
installed.

By default, if the `backend` keyword is missing, Everest will select the Dakota
backend.

The Dakota backend
------------------

The Dakota backend is the default choice for the optimization backend, it will
be selected when the `backend` keyword is not present, or if it is set to
`dakota`. Consult the documentation on the configuration file
(:ref:`cha_config`) for the algorithms that can be selected by the `algorithm`
keyword.

Additional options specific to Dakota can be passed as a list of strings using
the `options` keyword. The `backend_options` keyword is ignored by the Dakota
backend. The main differences between the Dakota `optpp_q_newton` and `conmin_mfd`
are listed in :ref:`cha_optimization_algorithms`.

.. important::

	For more information regarding specific optimizer settings please refer to the official `Dakota reference manual <https://dakota.sandia.gov/content/latest-reference-manual>`_.

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

The SciPy backend
-----------------

Everest supports algorithms from the SciPy Optimization package (`scipy.optimize
<https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html>`_). This
requires that SciPy is installed, otherwise Everest will raise an error if an
attempt is made to use this backend. Set the `backend` keyword to `scipy` to
select this backend, and set the `algorithms` keyword to one of the values
listed in the documentation of the configuration file (:ref:`cha_config`).

Additional options for the SciPy algorithms can be passed using the
`backend_options` keyword as a dictionary containing the option names and their
values. Consult the `scipy.optimize.minimize
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize>`_
documentation for the options supported by each algorithm. The `options` keyword
is ignored by the SciPy backend.

**Example**

.. code-block:: yaml

    optimization:
        backend: scipy
        algorithm: SLSQP
        convergence_tolerance: 0.001
        constraint_tolerance: 0.001
        perturbation_num: 7
        speculative: True
        backend_options:
            ftol: 1e-5
            disp: True
