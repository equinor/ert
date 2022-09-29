Queue system
============

The Queue System is responsible for scheduling, running, monitoring and acting
on a set of realizations.

Available queue systems
-----------------------

A number of queue systems are pre-installed with ERT.

- ``LOCAL``
- ``LSF``
- ``TORQUE``
- ``SLURM``

The ``LOCAL`` queue system will, unsurprisingly, run all computations locally,
whereas the remaining systems requires access to some external resource. Let's
first configure a local queue system.

Local
-----

Available local configuration options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: none

  QUEUE_SYSTEM LOCAL

which makes ERT use the ``LOCAL`` queue system, and

.. code-block:: none

  QUEUE_OPTION LOCAL MAX_RUNNING 50

which sets the ``MAX_RUNNING`` option for the ``LOCAL`` queue to ``50``. This
is the only *generic* option, i.e. an option that can be set for all queue
systems.

Creating a local configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's create ``local_queue.ert`` with the following content:

    .. literalinclude:: queue/local_queue.ert

In addition to this config, we'll also need a job config ``queue_test_job``:

    .. literalinclude:: queue/queue_test_job

As well as the actual job, ``queue_test_forward_model.py``:

    .. literalinclude:: queue/queue_test_forward_model.py
        :linenos:
        :language: python


Running ERT with this configuration, you can find the hostname of your machine
in the ``STDOUT`` of the job.

Note that running the *test experiment* will always run on the ``LOCAL`` queue,
no matter what your configuration says.

LSF
---

The LSF system is the most useful of the queue alternatives, and also the
alternative with most options. The most important options are related to how
ERT  should submit jobs to the LSF system. Essentially there are two methods
ERT can use when submitting jobs to the LSF system:

#. Workstations which have direct access to LSF can submit directly with
   no further configuration. This is the preferred solution, but unfortunately
   not very common.
#. Alternatively ERT can issue shell commands to ``bsub/bjobs/bkill`` to submit
   jobs. These shell commands can be issued on the current workstation, or
   alternatively on a remote workstation using ssh.

The main switch between alternatives 1 and 2 above is the ``LSF_SERVER``
option.

See also the complete :ref:`list of available LSF configuration options
<lsf_list_of_kwds>`.

Creating an LSF queue configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's create an ERT configuration that uses LSF. The following configuration
makes some assumptions:

- Password-less SSH access to the compute cluster;
- that the ``mr`` LSF queue exists;
- that ``job_dispatch.py`` is on the ``PATH`` for your user on the LSF server;
  and
- that the *runpath* (i.e. a folder with name ``queue_testing`` inside the
  current working directory) is accessible from the LSF server.

Note that ``queue_test_job`` and ``queue_test_forward_model.py`` remains the
same as before.

    .. literalinclude:: queue/lsf_queue.ert

For most users of ERT, most of the necessary LSF options have already been set
by the ``site-config``, which is a *site* wide configuration.

``STDOUT`` of the ``queue_test_job`` forward model should be something similar
to ``<be/st/tr/...>-lcn01-01-04``.

TORQUE
------

The Terascale Open-source Resource and QUEue Manager (*TORQUE*) queue system is
a distributed resource manager providing control over batch jobs and
distributed compute nodes.

The TORQUE system is the only available system on some clusters. The most
important options are related to how ERT should submit jobs to the TORQUE
system.

* Currently, the TORQUE option only works when the machine you are logged into
  have direct access to the queue system. ERT then submits directly with no
  further configuration.

The most basic invocation is in other words:

::

    QUEUE_SYSTEM TORQUE

See the :ref:`list of all TORQUE configuration options <torque_list_of_kwds>`.


SLURM
-----

Slurm is an open source queue system with many of the same capabilites as LSF.
The Slurm support in ERT assumes that the computer you are running on is part of
the Slurm cluster and no capabilities for ssh forwarding, shell to use and so on
is provided.

The Slurm support in ERT interacts with the Slurm system by issuing slurm
commands ``sbatch, sinfo, squeue`` and ``scancel`` and parsing the output from
these commands. By default the slurm driver will assume that the commands are in
``PATH`` - i.e. the command to submit will be the equivalent of:

::

     bash% sbatch submit_script.sh

But you can configure which binary should be used by using the ``QUEUE_OPTION
SLURM ..`` configuration command:

::

    QUEUE_OPTION SLURM SBATCH  /path/to/special/sbatch
    QUEUE_OPTION SLURM SINFO   /path/to/special/sinfo
    QUEUE_OPTION SLURM SQUEUEU /path/to/special/sqeueue
    QUEUE_OPTION SLURM SCANCEL /path/to/special/scancel


NUM_CPU
-------

The keyword ``NUM_CPU`` is a general keyword which is set straight in your
configuration file:

.. code-block:: none

  NUM_CPU 42

Its meaning varies depending on context. For LSF it equates to the ``-n``
parameter. See more here https://www.ibm.com/support/knowledgecenter/SSWRJV_10.1.0/lsf_command_ref/bsub.n.1.html.
E.g. ``NUM_CPU 10`` can be understood as a way for a job to make sure it can
execute on ``10`` processors. This means that a higher number may *increase*
wait times, since LSF might need to wait until resources are freed in order to
allocate 10 processors.

For TORQUE, it literally is a check that ``NUM_CPU`` is larger than the amount
of resources TORQUE wants to allocate (number_of_nodes \* cpus_per_node). See
:ref:`NUM_NODES|NUM_CPUS_PER_NODE <torque_nodes_cpus>` for details.

For the local queue system, ``NUM_CPU`` is ignored.
