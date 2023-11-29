
.. _forward_model_chapter:

Forward model
=============

In the context of uncertainty estimation and data assimilation,
a forward model refers to a predictive model that simulates how a system evolves
over time given certain inputs or intial conditions.
The model is called "forward" because it predicts the future state of the system based
on the current state and a set of input parameters.
The predictive model may include pre-processing and post-processing steps in addition
to the physics simulator itself.
In ERT, we think of a forward model as a sequence of jobs such as making directories,
copying files, executing simulators etc.

Consider a scenario in reservoir management.
Here, a forward model might encompass reservoir modeling software like RMS,
a fluid simulator like Eclipse or Flow, and custom jobs like relative permeability interpolation
and water saturation calculation.

To add a job to the forward model, use the :code:`FORWARD_MODEL` keyword.
Each :code:`FORWARD_MODEL` keyword instructs ERT to run a specific executable.
You can build a series of jobs by listing multiple 'FORWARD_MODEL' keywords.

An alternative to :code:`FORWARD_MODEL` is the :code:`SIMULATION_JOB` keyword,
which can also configure the forward model.
The difference lies in how these keywords pass command-line arguments to the final executable.

You can find all pre-configured jobs to define your forward models :ref:`here <Pre-configured jobs>`.
These jobs form the building blocks for your custom forward models in ERT.

.. _configure_own_jobs:

Configuring your own jobs
~~~~~~~~~~~~~~~~~~~~~~~~~

ERT imposes no restrictions on the programming language used to write a job.
The only requirement is that it should be an executable that can be run.
Consequently, it is possible to create a program or script performing any desired function,
and then have ERT run it as one of the jobs in the :code:`FORWARD_MODEL`.

However, for ERT to recognize a job, it must be installed. All predefined
jobs are already installed and may be invoked by using the
:code:`FORWARD_MODEL` keyword in the configuration file.
If you need to include a custom job, it must first be installed using :code:`INSTALL_JOB`,
as follows:

.. code-block:: bash

    INSTALL_JOB JOB_NAME JOB_CONFIG

In this command, JOB_NAME is a name of your choice that you can later use in
the ERT configuration file to call upon the job.
:code:`JOB_CONFIG` is a file that specifies the location of the executable
and provides rules for the behavior of any arguments.

By installing your own jobs in this way, you can extend the capabilities of ERT to meet your specific needs and scenarios.

.. code-block:: bash

    EXECUTABLE  path/to/program

    STDERR      prog.stderr      -- Name of stderr file (defaults to
                                 -- name_of_file.stderr.<job_nr>)
    STDOUT      prog.stdout      -- Name of stdout file (defaults to
                                 -- name_of_file.stdout.<job_nr>)
    ARGLIST     <ARG0> <ARG1>    -- A list of arguments to pass on to the
                                 --  executable

Note
____
When configuring ARGLIST for FORWARD_MODEL jobs it is not suitable to use
:code:`--some-option` for named options as it treated as a comment by the
configuration compiler. Single letter options, i.e. :code:`-s` are needed.

Invoking the job is then done by including it in the ert config:

.. code-block:: bash

    FORWARD_MODEL JOB_NAME(<ARG0>=3, <ARG1>="something")


Note that the following behaviour provides identical results:

.. code-block:: bash

    DEFINE <ARG0> 3
    FORWARD_MODEL JOB_NAME(<ARG1>="something")

see example :ref:`create_script`

.. _Pre-configured jobs:
.. ert_forward_model::
