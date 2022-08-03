.. _forward_model_chapter:

Forward model
=============

.. todo::
   List of all forward models and what they do

The ability to run arbitrary executables in a *forward model* is an absolutely
essential part of the ERT application. Schematically the responsability of the
forward model is to "transform" the input parameters to simulated results. The
forward model will typically contain a reservoir simulator like Eclipse or flow;
for an integrated modelling workflow it will be natural to also include
reservoir modelling software like rms. Then one can include jobs for special
modelling tasks like e.g. relperm interpolation or water saturation, or to
calculate dynamic results like seismic response for special purpose comparisons.


The `FORWARD_MODEL` in the ert configuration file
-------------------------------------------------

The traditional way to configure the forward model in ERT is through the keyword
:code:`FORWARD_MODEL`. Each :code:`FORWARD_MODEL` keyword will instruct ERT to run one
particular executable, and by listing several :code:`FORWARD_MODEL` keywords you can
configure a list of jobs to run.


The `SIMULATION_JOB` alternative
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to :code:`FORWARD_MODEL` there is an alternative keyword :code:`SIMULATION_JOB`
which can be used to configure the forward model. The difference between
:code:`FORWARD_MODEL` and :code:`SIMULATION_JOB` is in how commandline arguments are passed
to the final executable.


The runpath directory
---------------------

Default jobs
~~~~~~~~~~~~

It is quite simple to *install* your own executables to be used as forward model
job in ert, but some common jobs are distributed with the ert/libres
distribution.

see :doc:`../forward_models`


Jobs for geophysics
~~~~~~~~~~~~~~~~~~~


.. _configure_own_jobs:

Configuring your own jobs
~~~~~~~~~~~~~~~~~~~~~~~~~

ERT does not limit the type of programming language in which a job is written,
the only requirement is that it is an executable that can be run. It is
therefore possible to create a program, or a script, that does whatever the
user wishes, and then have ERT run it as one of the jobs in the
:code:`FORWARD_MODEL`.

A job must be `installed` in order for ERT to know about it. All predefined
jobs are already installed and may be invoked by including the
:code:`FORWARD_MODEL` keyword in the configuration file. Any other job must
first be installed with :code:`INSTALL_JOB` as such:

.. code-block:: bash

    INSTALL_JOB JOB_NAME JOB_CONFIG


The :code:`JOB_NAME` is an arbitrary name that can be used later in the ert
configuration file to invoke the job. The :code:`JOB_CONFIG` is a file that
specifies where the executable is, and how any arguments should behave.

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
configuration compiler. Single letter options, i.e. :code:`-s` as shown in the
examples in :doc:`../forward_models`, is needed.


Invoking the job is then done by including it in the ert config:

.. code-block:: bash

    FORWARD_MODEL JOB_NAME(<ARG0>=3, <ARG1>="something")


Note that the following behaviour provides identical results:

.. code-block:: bash

    DEFINE <ARG0> 3
    FORWARD_MODEL JOB_NAME(<ARG1>="something")

see example :ref:`create_script`

The `job_dispatch` executable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Interfacing with the cluster
----------------------------
