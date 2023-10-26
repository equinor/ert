The forward model
=================

Purpose
-------

The task of the forward model is to bring the model under study forward and by
doing so create its response to the given input. For an overview of how the
forward model fits into the larger picture of an ensemble we refer the reader
to :doc:`concepts`.

ERT forward models
------------------

Currently, an ERT forward model is a list of shell commands, called jobs, that
are executed sequentially. There is some additional context surrounding each
job that can be configured in the job configuration file, like environment
variables and default argument values. However, it quickly boils down to
sequential shell commands using the underlying file system for data
communication.

The input to a forward model consists of:

- a local disk area where the forward model is to be executed, the *runpath*
  of the forward model
- some of the parameters (:code:`GEN_KW` parameters) persisted as *JSON* in
  the root of the runpath in a file named :code:`parameters.json`.
- surface and field parameters (2D and 3D parameters) persisted directly into
  the reservoir model.
- A description of the forward model as *JSON* in the root of the runpath in a
  file named :code:`jobs.json`.
- The magic strings (string to string mapping for replacement) are being
  replaced in the runpath.

After this, the forward model is submitted to the queue system. In particular,
this entails that the script :code:`job_dispatch.py` is executed with the runpath of
the forward model as an argument. It will locate the :code:`jobs.json` file and
execute the forward model as prescribed. During execution the status of the
forward model is dumped to the :code:`status.json` file. It contains information
about whether each job was a success or failure, the start and end running
time, memory usage and so forth. In addition, each job's standard out and
standard error is piped to unique files for each job. Both the status
file and the log files are picked up again by the core for monitoring
purposes.

The run environment of the forward model using a plain ERT installation is the
same as the core. That entails that environment
variables are carried over and one assumes that the disk where the relevant
environment is installed is also available when running the forward model.
Furthermore, the system is configured in Equinor such that the named deploy
used by the core is persisted to disk and then picked up again on the forward
model side to ensure that upgrades are not impacting already launched runs.

After the forward model is completed, the overall status of the forward model is
signaled by either producing a file named :code:`OK` or one named :code:`ERROR` in the root
of the runpath. After this, response loading is initiated by the core. In
particular, summary data is loaded from the configured :code:`ECLBASE` and in
addition the various :code:`GEN_DATA` etc. responses are loaded from their
configured files. If the loading of responses is also successful, the forward
model in its entirety is deemed successful by the core.