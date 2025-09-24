.. _queue-system-chapter:

Queue system
============

The queue system is responsible for scheduling, running, monitoring and acting
on a set of realizations.

A number of queue systems are pre-installed with ERT:

- ``LOCAL`` — run locally, on your machine. :ref:`Details.<local-queue>`
- ``LSF`` — send computation to an LSF cluster. :ref:`Details.<lsf-systems>`
- ``TORQUE`` — send computation to a TORQUE or PBS cluster. :ref:`Details.<pbs-systems>`
- ``SLURM`` — send computation to a Slurm cluster. :ref:`Details.<slurm-systems>`

Select the system using the ``QUEUE_SYSTEM`` keyword. For example, the
following line in your ERT configuration file specifies that ERT should use the
``LOCAL`` system::

  QUEUE_SYSTEM LOCAL

This page documents the configuration options available for each queue system.
Some of the options apply to all systems, others only to specific queue systems.


.. _all-queues:

Options that affect all queue systems
-------------------------------------

In addition to the queue-specific settings, the following options affect
all queue systems. These are documented in :ref:`ert_kw_full_doc`.

* ``JOB_SCRIPT`` — see :ref:`List of keywords<job_script>`
* ``MAX_RUNNING`` — see :ref:`List of keywords<max_running>`
* ``MAX_RUNTIME`` — see :ref:`List of keywords<max_runtime>`
* ``MAX_SUBMIT`` — see :ref:`List of keywords<max_submit>`
* ``NUM_CPU`` — see :ref:`List of keywords<num_cpu>`
* ``STOP_LONG_RUNNING`` — see :ref:`List of keywords<stop_long_running>`
* ``SUBMIT_SLEEP`` — see :ref:`List of keywords<queue_option_submit_sleep>`


.. _local-queue:

On Crash or Exit
----------------

Realizations that were submitted before the early exit will keep on running.
Results in a runpath can be loaded manually if enough realizations completed.
Normally ert will do this automatically at the end of each iteration.
See also :ref:`restarting_es-mda`.

LOCAL queue
-----------

Let's create ``local_queue.ert`` with the following content:

.. literalinclude:: ../queue/local_queue.ert

In addition to this config, we'll also need a forward model config
``QUEUE_TEST``:

.. literalinclude:: ../queue/QUEUE_TEST

As well as the actual forward model, ``queue_test_forward_model.py``:

.. literalinclude:: ../queue/queue_test_forward_model.py
  :language: python

Running ERT with this configuration, you can find the hostname of your machine
in the ``STDOUT`` of the run.

Note that running the *test experiment* will always run on the ``LOCAL`` queue,
no matter what your configuration says.

There is only one queue option for the local queue system: ``MAX_RUNNING``.

.. _local_max_running:
.. topic:: MAX_RUNNING

  The queue option MAX_RUNNING controls the maximum number of simultaneously
  submitted and running realizations, where ``n`` is a positive integer::

    QUEUE_OPTION LOCAL MAX_RUNNING n

  If ``n`` is zero (the default), then there is no limit, and all realizations
  will be started as soon as possible.


.. _lsf-systems:

LSF systems
-----------

IBM's `Spectrum LSF software <https://www.ibm.com/docs/en/spectrum-lsf/10.1.0?topic=overview-lsf-introduction>`_
is a common queue management system in high-performance computing environments.

The following example configuration makes some assumptions:

- Passwordless ``ssh`` access to the compute cluster.
- The ``mr`` LSF queue exists (check available queues with ``bqueues``).
- The *runpath* (i.e. a folder with name ``queue_testing`` inside the
  current working directory) is accessible from the LSF server.

Note that the ``QUEUE_TEST`` forward model config file and
``queue_test_forward_model.py`` remain the same as before.

.. literalinclude:: ../queue/lsf_queue.ert

It is possible to set LSF options in the ``site-config``, which is a *site wide*
configuration that affects all users.

The following is a list of available LSF configuration options:

.. _bsub_cmd:
.. topic:: BSUB_CMD

  The submit command. Default: ``bsub``. To change it::

    QUEUE_OPTION LSF BSUB_CMD command

.. _bjobs_cmd:
.. topic:: BJOBS_CMD

  The queue query command. Default: ``bjobs``. To change it::

    QUEUE_OPTION LSF BJOBS_CMD command

.. _bkill_cmd:
.. topic:: BKILL_CMD

  The kill command. Default: ``bkill``. To change it::

    QUEUE_OPTION LSF BKILL_CMD command

.. _bhist_cmd:
.. topic:: BHIST_CMD

  The queue history command. Default: ``bhist``. To change it::

    QUEUE_OPTION LSF BHIST_CMD command

.. _queue_option_submit_sleep:
.. topic:: SUBMIT_SLEEP

  Determines for how long in seconds the system will sleep between submitting jobs.
  Default: ``0``. To change it to 1 second::

    QUEUE_OPTION LSF SUBMIT_SLEEP 1

.. _lsf_queue:
.. topic:: LSF_QUEUE

  The name of the LSF queue you wish to send simulations to. The parameter
  will be passed as ``bsub -q name_of_queue`` (assuming ``bsub`` is the
  submit command you are using). `Docs. <https://www.ibm.com/support/knowledgecenter/SSWRJV_10.1.0/lsf_command_ref/bsub.q.1.html>`__
  Usage::

    QUEUE_OPTION LSF LSF_QUEUE name_of_queue

.. _lsf_resource:
.. topic:: LSF_RESOURCE

  A resource requirement string describes the resources that a job needs.
  LSF uses resource requirements to select hosts for remote execution and
  job execution. Resource requirement strings can be simple (applying to the
  entire job) or compound (applying to the specified number of slots).
  The value passed does not use units and depends on the cluster's configuration,
  so follow up with cluster administrator to find out what the set unit is.
  `Docs. <https://www.ibm.com/support/knowledgecenter/SSWRJV_10.1.0/lsf_admin/res_req_strings_about.html>`__
  Passed as the ``-R`` option to ``bsub``. For example, this will
  request approximately 15 gigabytes when the default unit is megabytes::

    QUEUE_OPTION LSF LSF_RESOURCE rusage[mem=15000]

.. _project_code:
.. topic:: PROJECT_CODE

  String identifier used to map hardware resource usage to a project or account.
  The project or account does not have to exist.

  Equates to the ``-P`` parameter for e.g. ``bsub``.
  `See docs. <https://www.ibm.com/support/knowledgecenter/SSWRJV_10.1.0/lsf_command_ref/bsub.__p.1.html>`__
  For example, to register jobs in the ``foo`` project::

    QUEUE_OPTION LSF PROJECT_CODE foo

  If the option is not set in the config file and the forward model section contains
  any of the following simulator jobs [RMS, FLOW, ECLIPSE100, ECLIPSE300]
  a default will be set.::

    FORWARD_MODEL RMS <args>
    FORWARD_MODEL ECLIPSE100 <args>

  This will set the PROJECT_CODE option to ``rms+eclipse100``

.. _exclude_host:
.. topic:: EXCLUDE_HOST

  Comma separated list of hosts to be excluded. The LSF system will pass this
  list of hosts to the ``-R`` argument of e.g. ``bsub`` with the criteria
  ``hname!=<exluded_host_1>``. For example::

    QUEUE_OPTION LSF EXCLUDE_HOST host1,host2

.. _lsf_max_running:
.. topic:: MAX_RUNNING

  The queue option MAX_RUNNING controls the maximum number of simultaneous jobs
  submitted to the queue when using (in this case) the LSF queue system, where ``n``
  is a positive integer::

    QUEUE_OPTION LSF MAX_RUNNING n

  If ``n`` is zero (the default), then it is set to the number of realizations.


.. _pbs-systems:

TORQUE and PBS systems
----------------------

`TORQUE <https://adaptivecomputing.com/cherry-services/torque-resource-manager/>`_
is a distributed resource manager providing control over batch jobs and
distributed compute nodes; it implements the API of the Portable Batch System
(PBS), so is compatible with systems using `OpenPBS <https://openpbs.org/>`_
or Altair's `PBS Professional <https://altair.com/pbs-professional>`_.

ERT offers several options specific to the TORQUE/PBS queue system, controlling
how it submits jobs. Currently, the option only works when the machine
you are logged into has direct access to the queue system. ERT then submits
directly with no further configuration.

To instruct ERT to use a TORQUE/PBS queue system, use the following
configuration::

  QUEUE_SYSTEM TORQUE

The following is a list of all queue-specific configuration options:

.. _torque_sub_stat_del_cmd:
.. topic:: QSUB_CMD, QSTAT_CMD, QDEL_CMD

  By default ERT will use the shell commands ``qsub``, ``qstat`` and ``qdel``
  to interact with the queue system, i.e. whatever binaries are first in your
  PATH will be used. For fine grained control of the shell based submission
  you can tell ERT which programs to use::

    QUEUE_SYSTEM TORQUE
    QUEUE_OPTION TORQUE QSUB_CMD /path/to/my/qsub
    QUEUE_OPTION TORQUE QSTAT_CMD /path/to/my/qstat
    QUEUE_OPTION TORQUE QDEL_CMD /path/to/my/qdel

.. _torque_queue:
.. topic:: QUEUE

  The name of the queue you are running simulations in. Example::

    QUEUE_OPTION TORQUE QUEUE name_of_queue

.. _torque_cluster_label:
.. topic:: CLUSTER_LABEL

  The name of the cluster you are running simulations in. This
  might be a label (several clusters), or a single one, as in this example::

    QUEUE_OPTION TORQUE CLUSTER_LABEL baloo

.. _torque_max_running:
.. topic:: MAX_RUNNING

  The queue option MAX_RUNNING controls the maximum number of simultaneous jobs
  submitted to the queue when using the queue system, where ``n`` is a positive
  integer::

    QUEUE_OPTION TORQUE MAX_RUNNING n

  If ``n`` is zero (the default), then it is set to the number of realizations.

.. _torque_keep_qsub_output:
.. topic:: KEEP_QSUB_OUTPUT

  Sometimes the error messages from ``qsub`` can be useful, if something is
  seriously wrong with the environment or setup. To keep this output (stored
  in your home folder), use this::

    QUEUE_OPTION TORQUE KEEP_QSUB_OUTPUT 1

.. _torque_submit_sleep:
.. topic:: SUBMIT_SLEEP

  To avoid stressing the TORQUE/PBS system you can instruct the driver to sleep
  for every submit request. The argument to the SUBMIT_SLEEP is the number of
  seconds to sleep for every submit, which can be a fraction like 0.5::

    QUEUE_OPTION TORQUE SUBMIT_SLEEP 0.5

.. _torque_project_code:
.. topic:: PROJECT_CODE

  String identifier used to map hardware resource usage to a project or account.
  The project or account does not have to exist.

  Equates to the ``-A`` parameter for``qsub``
  `see docs. <https://www.jlab.org/hpc/PBS/qsub.html>`__
  For example, to register jobs under the ``foo`` account::

    QUEUE_OPTION TORQUE PROJECT_CODE foo

  If the option is not set in the config file and the forward model section contains
  any of the following simulator jobs [RMS, FLOW, ECLIPSE100, ECLIPSE300]
  a default will be set.::

    FORWARD_MODEL RMS <args>
    FORWARD_MODEL ECLIPSE100 <args>

  This will set the PROJECT_CODE option to ``rms+eclipse100``

.. _slurm-systems:

Slurm systems
-------------

`Slurm <https://slurm.schedmd.com/>`_ is an open source queue system with many
of the same capabilites as LSF. The Slurm support in ERT assumes that the
computer you are running on is part of the Slurm cluster and no capabilities
for ``ssh`` forwarding, shell to use and so on is provided.

The Slurm support in ERT interacts with the Slurm system by issuing ``sbatch``,
``sinfo``, ``squeue`` and ``scancel`` commands, and parsing the output from
these commands. By default the Slurm driver will assume that the commands are in
``PATH``, i.e. the command to submit will be the equivalent of::

  bash% sbatch submit_script.sh

But you can configure which binary should be used by using the
``QUEUE_OPTION SLURM`` configuration command, for example::

  QUEUE_OPTION SLURM SBATCH  /path/to/special/sbatch
  QUEUE_OPTION SLURM SINFO   /path/to/special/sinfo
  QUEUE_OPTION SLURM SQUEUE  /path/to/special/squeue
  QUEUE_OPTION SLURM SCANCEL /path/to/special/scancel

The Slurm queue managing tool has a very fine grained control. In ERT
only the most necessary options have been added.

.. _slurm_sbatch:
.. topic:: SBATCH

  Command used to submit the jobs, default ```sbatch``. To change the executable
  to, for example, ``/opt/bin/sbatch``, do this::

    QUEUE_OPTION SLURM SBATCH /opt/bin/sbatch

.. _slurm_scancel:
.. topic:: SCANCEL

  Command used to cancel the jobs, default ``scancel``.

.. _slurm_scontrol:
.. topic:: SCONTROL

  Command to modify configuration and state, default ``scontrol``.

.. _slurm_sacct:
.. topic:: SACCT

  Command used when ``scontrol`` fails, default ``sacct``.

.. _slurm_squeue:
.. topic:: SQUEUE

  Command to view information about the queue, default ``squeue``.

.. _slurm_partition:
.. topic:: PARTITION

  Partition/queue in which to run the jobs, for example to use ``foo``::

    QUEUE_OPTION SLURM PARTITION foo

.. _slurm_squeue_timeout:
.. topic:: SQUEUE_TIMEOUT

  Specify timeout in seconds used when querying for status of the jobs
  while running. For example::

    QUEUE_OPTION SLURM SQUEUE_TIMEOUT 10

.. _slurm_smax_runtime:
.. topic:: MAX_RUNTIME

  Specify the maximum runtime in seconds for how long a job can run, for
  example::

    QUEUE_OPTION SLURM MAX_RUNTIME 100

.. _slurm_include_host:
.. topic:: INCLUDE_HOST

  Specific host names to use when running the jobs. It is possible to add multiple
  hosts separated by space or comma in one option call, e.g.::

    QUEUE_OPTION SLURM INCLUDE_HOST host1,host2

.. _slurm_exclude_host:
.. topic:: EXCLUDE_HOST

  Specific host names to exclude when running the jobs. It is possible to add multiple
  hosts separated by space or comma in one option call, e.g.::

    QUEUE_OPTION SLURM EXCLUDE_HOST host3,host4

.. _max_running_slurm:
.. topic:: MAX_RUNNING

  The queue option MAX_RUNNING controls the maximum number of simultaneous jobs
  submitted to the queue when using the queue system, where ``n`` is a positive
  integer::

    QUEUE_OPTION SLURM MAX_RUNNING n

  If ``n`` is zero (the default), then it is set to the number of realizations.

.. _slurm_project_code:
.. topic:: PROJECT_CODE

  String identifier used to map hardware resource usage to a project or account.
  The project or account does not have to exist.

  Equates to the ``-A`` parameter for ``sbatch``
  `see docs. <https://slurm.schedmd.com/sbatch.html#OPT_account>`__
  For example, to register jobs under the ``foo`` account::

    QUEUE_OPTION SLURM PROJECT_CODE foo

  If the option is not set in the config file and the forward model section contains a
  any of the following simulator jobs [RMS, FLOW, ECLIPSE100, ECLIPSE300]
  a default will be set.::

    FORWARD_MODEL RMS <args>
    FORWARD_MODEL ECLIPSE100 <args>

  This will set the PROJECT_CODE option to ``rms+eclipse100``

GENERIC queue options
---------------------

There are a number of queue options valid for all queue systems and for those we can use
the `GENERIC` keyword. ::

    QUEUE_SYSTEM LSF
    QUEUE_OPTION GENERIC MAX_RUNNING 10
    QUEUE_OPTION GENERIC SUBMIT_SLEEP 2

Is equivalent to::

    QUEUE_SYSTEM LSF
    QUEUE_OPTION LSF MAX_RUNNING 10
    QUEUE_OPTION LSF SUBMIT_SLEEP 2
