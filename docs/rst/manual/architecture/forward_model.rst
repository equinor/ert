The forward model
=================

Purpose
-------

The task of the forward model is to bring the model under study forward and by
doing so create its response to the given input. For an overview of how the
forward model fits into the larger picture of an ensemble we refer the reader
to :doc:`concepts`.

ERT forward models 1.0
----------------------

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

ERT forward models 2.0
----------------------

The main goal of the future ERT forward model is to make ERT more robust and
less aware of the underlying infrastructure. And furthermore, to allow for more
advanced tooling for executing the forward model and for treating the data
going in and out of the forward model.

Steps
~~~~~

The basic building block of a forward model is to be a *step*. A step will
expose little internal structure, but will have an explicit relationship to the
run environment and input and output data. Furthermore, it will be typed
according to its data protocol and run environment assumptions.

Scripts
"""""""

The first step type that will be implemented is *script*. A script is a step
for which one is provided with a local file system that one can utilize, where
input data is persisted to disk before the script is launched and output data
can either be picked up from the disk or published via an API (reporter). The
run environment will be explicitly configured by the user. As such a script is
a maturation of *forward model 1.0*. We must accept that relevant tooling will
require a disk system for a long time and we should facilitate this while
allowing parts of the forward model tooling to mature faster.

All input to the script is to be persisted in general data formats. There is
however need for some configurability on this. In particular, while some input
is naturally persisted as JSON, larger data sets like surfaces and fields will
need a different representation. Candidates like *csv*, *npy* and *hdf5*
should be considered. Support for rendering templates and persisting values in
reservoir models (and other domain and tool specific formats) should be the
responsibility of the script.

Commands that can be executed within a script should be possible to install,
together with its corresponding documentation and command configuration in the
configuration system of ERT. Then, the logic of the step can be written in a
shell like fashion:
::

    template_render -i magic_strings -o simulation_deck -t simulation_template.jinja2
    super_simulator simulation_deck --fast

A script can produce output in two different manners. Either by the script
creating a file on disk that ERT is configured to pick up, or by utilizing the
reporter. In both cases the focus should be on supporting standard data formats
in the same manner as for input data. If responses are written to disk, again
formats like *JSON*, *csv*, *npy* and *hdf5* should be considered. When using
the reporter an API for publishing data on corresponding standard formats
should be available for the commands so that they can publish the data without
caring about the disk.

The environment for which the script is executed in should be detached from
that of the core. A first attempt on this could be made utilising virtual
environments; where a base environment is specified and then we support
extending the environment by pip installation and transporting scripts that can
be moved as a single file from the location of the ERT core to the location
for which the scrip is executed. In Equinor this could then be based on komodo
environments, with additional packages installed via pip. Later, we should look
into the need for further extending the capabilities of configuring the run
environment. In particular, support for containers along the lines of
`Slurm <https://slurm.schedmd.com/containers.html>`_ might be a reasonable
approach to this.

The status and progress of the script should be published by the script
executor via the reporter. If a finer granularity of status reporting is wanted
we should make it possible for the individual commands to publish progress via
the reporter as well. Furthermore, starting to stream data to the core before a
command is completed is a frequently requested vision among our users.

Functions
"""""""""

Later, we should aim for implementing support for function steps. Where a
functional action to the data can be implemented in for instance Python and
applied to the data without touching disk.

Stages
------

The role of a stage is to combine multiple steps (if needed) into a logical
unit where you can reason about the data flow. This allows you to combine step
types into a unit executing a certain task. Perhaps your simulator still
depends on file system to run, but some post-processing is better implemented
as a function on top of the results? Or perhaps one step requires a very
specific type of hardware, while other parts can be executed on a vanilla
machine. It is important that a stage supports a sufficiently rich language for
referencing input and output data. In particular, we must avoid that the names
of output data in one step must align with the naming of input data in another
step. It is the role of a stage to facilitate the data flow between these
stages without the implementation of one leaking into the implementation of the
other.

Output of a stage should be guaranteed to be persisted such that it can be
restarted without having to rerun the stages before it. This will allow for an
iterative approach to modelling in ERT without enforcing reruns in an
experimental phase. It will also allow for restarts if the infrastructure goes
down or bugs occur. And it will allow for building new experiments on top of
existing ones.

An important thing to notice is that with a sufficiently rich data flow
language there is no need to assume that a stage is sequential anymore. In
particular, different stages might be executed in parallel if neither of them
depends on the data produced by the others.

Forward model as a stage
------------------------

By considering a stage also a step, one can combine stages again within a
stage. And in such a world a forward model would be nothing but the root
stage.

Data provenance
---------------

By having a more explicit relationship to the data flow between different steps
ERT can facilitate data provenance and lineage during the ensemble modelling
process. This will enable functionality as partial restarts, iterations without
rerunning the full forward model etc. It will also enable visualisation of the
data flow and revisions of previous experiments and its results. This will
definitively be an iterative learning process, but it all starts with a more
explicit relationship to the data flow.
