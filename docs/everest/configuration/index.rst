.. _cha_config:

*********************
EVEREST configuration
*********************

.. toctree::
    :hidden:

    reference

EVEREST is configured via a yaml file, using a set of pre-defined keywords that
are described in more detail in the section :ref:`cha_config_reference`.

.. _config_variables:

Configuration variables
-----------------------

In addition to the standard yaml syntax, EVEREST also supports variables that
are replaced with their value when referred in the following way:
``r{{variable}}``. For instance in the following snippet, the variable ``tol``
is replaced by its value:

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

2. EVEREST pre-defines the following variables:

   ``realization``
       Evaluates to the realization ID during execution of the forward model.

   ``configpath``
       The fully qualified path to the directory that contains the configuration
       file.

   ``runpath_file``
       Evaluates to "<RUNPATH_FILE>", the fully qualified name of the runpath
       file that is saved before each batch is started.

   ``eclbase``
       Evaluates to "<ECLBASE>", used by ERT to set the basename for
       ECLIPSE simulations.

   These variables do not need to be defined by the user, although their values
   can be overridden in the ``definitions`` section. However, this is not
   recommended for the ``realization`` entry, and EVEREST will produce a warning
   when this is attempted.

3. Variables with a name of the form ``os.ENVIRONMENT_VARIABLE_NAME`` can be
   used to access environment variables. For instance, the variable
   ``r{{os.HOSTNAME}}`` will be replaced by the contents of the environment
   variable ``HOSTNAME``.

.. note::
    Variables are a distinct feature from the yaml keywords defined in section
    :ref:`cha_config_reference`. The final yaml file used by EVEREST is produced
    by pre-processing the config file to replace all variables with their value.
    It is possible to define variables that have the same name as a keyword, but
    this should be done sparingly to avoid confusion.
