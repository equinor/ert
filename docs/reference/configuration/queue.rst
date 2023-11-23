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

* ``MAX_SUBMIT`` — see :ref:`List of keywords<max_submit>`
* ``MAX_RUNTIME`` — see :ref:`List of keywords<max_runtime>`
* ``STOP_LONG_RUNNING`` — see :ref:`List of keywords<stop_long_running>`
* ``JOBNAME`` — see :ref:`List of keywords<jobname>`
* ``NUM_CPU`` — see :ref:`List of keywords<num_cpu>`


.. _local-queue:

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

The ``LOCAL`` queue system does not have any queue options. See :ref:`all-queues`.


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

.. _bjobs_timeout:
.. topic:: BJOBS_TIMEOUT

  Determines how long-lived the job cache is. Default: ``0`` (no cache).
  To change it to 60 s::

    QUEUE_OPTION LSF BJOBS_TIMEOUT 60

.. _submit_sleep:
.. topic:: SUBMIT_SLEEP

  Determines for how long the system will sleep between submitting jobs.
  Default: ``0``. To change it to 1 s::

    QUEUE_OPTION LSF SUBMIT_SLEEP 1

.. _lsf_server:
.. topic:: LSF_SERVER

  This options tells ERT which server should be used when submitting.
  So when your configuration file has the setting::

    QUEUE_OPTION LSF LSF_SERVER be-grid01

  ERT will use ``ssh`` to submit your jobs using shell commands on the server
  ``be-grid01``. For this to work you must have passwordless ``ssh`` to the
  server.

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
  `Docs. <https://www.ibm.com/support/knowledgecenter/SSWRJV_10.1.0/lsf_admin/res_req_strings_about.html>`__
  Passed as the ``-R`` option to ``bsub``. For example::

    QUEUE_OPTION LSF LSF_RESOURCE rusage[mem=512MB:swp=1GB]

.. _lsf_rsh_cmd:
.. topic:: LSF_RSH_CMD

  This option sets the remote shell command, which defaults to ``/usr/bin/ssh``.
  To use another command, pass the full path::

    QUEUE_OPTION LSF LSF_RSH_CMD /opt/bin/ssh

.. _lsf_login_shell:
.. topic:: LSF_LOGIN_SHELL

  Equates to the ``-L`` parameter of e.g. ``bsub``.
  Useful if you need to force the ``bsub`` command to use e.g. ``/bin/csh``.
  `Docs <https://www.ibm.com/support/knowledgecenter/en/SSWRJV_10.1.0/lsf_command_ref/bsub.__l.1.html>`__
  For example::

    QUEUE_OPTION LSF LSF_LOGIN_SHELL /bin/csh

.. _project_code:
.. topic:: PROJECT_CODE

  Equates to the ``-P`` parameter for e.g. ``bsub``.
  `See docs. <https://www.ibm.com/support/knowledgecenter/SSWRJV_10.1.0/lsf_command_ref/bsub.__p.1.html>`_
  For example, to register jobs in the ``foo`` project::

    QUEUE_OPTION LSF PROJECT_CODE foo

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

.. _torque_qstat_options:
.. topic:: QSTAT_OPTIONS

  Options to be supplied to the ``qstat`` command. This defaults to :code:`-x`,
  which tells the ``qstat`` command to include exited processes.

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

.. _torque_nodes_cpus:
.. topic:: NUM_NODES, NUM_CPUS_PER_NODE

  When using TORQUE/PBS systems, you can specify how many nodes a single job should
  use, and how many CPUs per node. These options are called ``NUM_NODES`` and
  ``NUM_CPUS_PER_NODE``. The default setup in ERT will use one node and
  one CPU.

  If the numbers specified is higher than supported by the cluster (e.g. use 32
  CPUs, but no node has more than 16), the job will not start.

  If you wish to increase this number, the program running will usually also
  have to be told to correspondingly use more processing units (e.g. for
  ECLIPSE, use the keyword ``PARALLEL``).

  The following should allow 3 × 8 = 24 CPUs for processing realizations::

    QUEUE_SYSTEM TORQUE
    QUEUE_OPTION TORQUE NUM_NODES 3
    QUEUE_OPTION TORQUE NUM_CPUS_PER_NODE 8

.. _torque_memory_per_job:
.. topic:: MEMORY_PER_JOB

  You can specify the amount of memory you will need for running your
  job. This will ensure that not too many jobs will run on a single
  shared memory node at once, possibly crashing the compute node if it
  runs out of memory.

  You can get an indication of the memory requirement by watching the
  course of a local run using the ``htop`` utility. Whether you should set
  the peak memory usage as your requirement or a lower figure depends on
  how simultaneously each job will run.

  The option to be supplied will be used as a string in the ``qsub``
  argument. You must specify the unit, either ``gb`` or ``mb`` as in
  the example::

    QUEUE_OPTION TORQUE MEMORY_PER_JOB 16gb

  By default, this value is not set.

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

.. _torque_queue_query_timeout:
.. topic:: QUEUE_QUERY_TIMEOUT

  The driver allows the backend TORQUE/PBS system to be flaky, i.e. it may
  intermittently not respond and give error messages when submitting jobs
  or asking for job statuses. The timeout (in seconds) determines how long
  ERT will wait before it will give up. Applies to job submission (``qsub``)
  and job status queries (``qstat``). Default is 126 seconds.

  ERT will do exponential sleeps, starting at 2 seconds, and the provided
  timeout is a maximum. Let the timeout be sums of series like 2+4+8+16+32+64
  in order to be explicit about the number of retries. Set to zero to disallow
  flakyness, setting it to 2 will allow for one re-attempt, and 6 will give two
  re-attempts. Example allowing six retries::

    QUEUE_OPTION TORQUE QUEUE_QUERY_TIMEOUT 254


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

.. _slurm_memory:
.. topic:: MEMORY

  Memory (in MiB) required per node, for example::

    QUEUE_OPTION SLURM MEMORY 16000

.. _slurm_memory_per_cpu:
.. topic:: MEMORY_PER_CPU

  Memory (in MiB) required per allocated CPU, for example::

    QUEUE_OPTION SLURM MEMORY_PER_CPU 4000

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

    QUEUE_OPTION TORQUE MAX_RUNNING n

  If ``n`` is zero (the default), then it is set to the number of realizations.
