.. _cha_config:

*********************
Everest configuration
*********************

.. toctree::
    :hidden:

    reference

Everest is configured via a yaml file, using a set of pre-defined keywords that
are described in more detail in the section :ref:`cha_config_reference`.

In addition to the standard yaml syntax, Everest also supports the use of
variables that are replaced with their value when referred in the following
way: ``r{{variable}}``. For instance in the following snippet, the variable
``tol`` is replaced by its value:

.. code-block:: yaml

    optimization:
        algorithm: optpp_q_newton
        convergence_tolerance: r{{tol}}

The value of a variable can be set in three different ways:

1. In the ``definitions`` section in the yaml file. For instance, to define a
   variable ``tol`` with a value of 0.0001, include this in the ``definitions``
   section:

   .. code-block:: yaml

       definitions:
           tol: 0.0001

2. Everest pre-defines the following variables:

   .. code-block:: yaml

      realization: <GEO_ID>
      configpath: <CONFIG_PATH>
      runpath_file: <RUNPATH_FILE>
      eclbase: <ECLBASE>

   These variables do not need to be defined by the user, although their values
   can be overridden in the ``definitions`` section. However, this is not
   recommended for the ``realization`` entry, and Everest will produce a warning
   when this is attempted.

3. Variables with a name of the form ``os.ENVIRONMENT_VARIABLE_NAME`` can be used to access
   the values of environment variables. For instance, the variable
   ``r{{os.HOSTNAME}}`` will be replaced by the contents of the environment
   variable ``HOSTNAME``.

.. note::
    Variables are a distinct feature from the yaml keywords defined in section
    :ref:`cha_config_reference`. The final yaml file used by Everest is produced
    by pre-processing the config file to replace all variables with their value.
    It is possible to define variables that have the same name as a keyword, but
    this should be done sparingly to avoid confusion.


.. _cha_config_reference:
