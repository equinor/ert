Running simulations - the Forward Model
=======================================

The ability to run arbitrary executables in a *forward model* is an absolutely
essential part of the ERT application. Schematically the responsability of the
forward model is to "transform" the input parameters to simulated results. The
forward model will typically contain a reservoir simulator like Eclipse or flow;
for an integrated modelling workflow it will be natural to also include
reservoir modelling software like rms. Then one can include jobs for special
modelling tasks like e.g. relperm interpolation or water saturation, or to
calculate dynamic results like seismic response for special purpose comparisons.



The forward model in the ert configuration file
-----------------------------------------------

The traditional way to configure the forward model in ERT is through the keyword
`FORWARD_MODEL`. Each `FORWARD_MODEL` keyword will instruct ERT to run one
particular executable, and by listing several `FORWARD_MODEL` keywords you can
configure a list of jobs to run.


The `SIMULATION_JOB` alternative
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to `FORWARD_MODEL` there is an alternative keyword `SIMULATION_JOB`
which can be used to configure the forward model. The difference between
`FORWARD_MODEL` and `SIMULATION_JOB` is in how commandline arguments are passed
to the final executable.



The runpath directory
---------------------

Default jobs
~~~~~~~~~~~~

It is quite simple to *install* your own executables to be used as forward model
jobs in ert, but some common jobs are distributed with the ert/libres
distribution.

Reservoir simulation: eclipse
.............................

Reservoir modelling: RMS
........................

File system utilities
.....................

Jobs for geophysics
~~~~~~~~~~~~~~~~~~~



Configuring your own jobs
~~~~~~~~~~~~~~~~~~~~~~~~~

The `job_dispatch` executable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Interfacing with the cluster
----------------------------

