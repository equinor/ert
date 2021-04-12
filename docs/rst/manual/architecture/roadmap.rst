.. _Roadmap:

Roadmap
=======

The ERT development project
---------------------------

ERT has been made the official history matching tool in Equinor and as part of
that decision the ERT Development Project was initialised. There are three main
pillars of this project, being *simplification*, *MCMC support* and *data
analysis*.

Simplification
~~~~~~~~~~~~~~

We are to lower the technical burden put on ERT users. The domain of ERT is
and always will be complex, but right now it is also extremely complicated to
operate ERT. We will address this by:

- a making a new yaml based configuration system,
- implement stronger configuration validation,
- have message and status propagation from the different components of ERT to the user,
- improved documentation and
- tooling to assist the user organise their experiments.

MCMC support
~~~~~~~~~~~~

We are to implement the Monte Carlo Markov Chain algorithm first implemented in the internal tool Basra.

Data analysis
~~~~~~~~~~~~~

We are to aid the users in their data analysis by both providing a
visualisation solution of the ensemble data, as well as visual support to
interact with and edit new experiments. This will not be a solution that will
cater all usage and needs, but will instead aim at providing an out-of-the-box
solution that will cover 80% of the usage. Hence, the remaining 20% is left for
specialised tools.

Configuration system and projects
---------------------------------

We will implement support for a new yaml based configuration system with
validation. In addition, we plan to support a project structure that allows for
a project specific configuration, with multiple experiments spawned from this
common configuration file. Each experiment will be defined by a separate
configuration file and will either carry out an `ensemble experiment`, `history
matching` or `sensitivity analysis`. In addition, there will be support for
internalising (and exporting) resources in such a way that observations (and
other relevant data) can be stored, edited and persisted.

Messages and feedback
---------------------

Support for passing messages regarding the status for the forward models, the
workflows, algorithms and other components of ERT to the user of ERT via an API
such that the choice of presentation can be taken further up in the call stack.
This also encompasses failures due to invalid configuration or other problems.
There should be established message contracts so that external, authenticated
parties also can consume the messages.

Forward models
~~~~~~~~~~~~~~

Currently this is done by the jobs in the forward model logging to files. These
should probably be propagated by the :code:`job_dispatch` to the rest of the system
via a message passing system.

Workflows
~~~~~~~~~

Workflows should be provided with an API for passing messages giving updates
regarding its status and expect the messages to be presented to the user.

Queue system
~~~~~~~~~~~~

The queue system is to facilitate running both forward models and workflows and
should propagate summary messages regarding its status via an established
message passing system.

Algorithms
~~~~~~~~~~

Algorithms should also have the possibility to send messages that it can expect
to be forwarded to the user.

Visualisation
-------------

Implement support for data visualisation of relevant data to help users carry
out good craftsmanship while using ERT. We should support plotting outputted
data from our forward models, together with observations, as well as
algorithmic data as misfit of different observations etc.

Observation analysis
--------------------

Users should be able to activate, deactivate, scale, group observations and
correlate them.

Improved documentation
----------------------

This work is twofold. First, we are to improve the current documentation
further. Second, we are to thouroughly document new features as the system
changes.

Data storage
------------

The data storage solution in ERT is to be improved significantly. By taking
ownership of data like observations (see the project section above) and using a
standard database solution we aim at providing a robust database that users can
trust their data with and that allows them to compare vastly different
experiments. There are two types of data that is to be supported; recognised
and unrecognised data. The recognisable data can be parsed by ERT and will be
possible to visualise, feed into algorithms etc. In other words data where
ERT can extract numerical data. In addition, we will need to support storing
unrecognised data that ERT has no knowledge about, but that ERT still can
store and serve to the user when necessary.

API
~~~

The storage solution is to have a well defined and documented API that allows
for project inspection and retrieval of all relevant data.

Data types
~~~~~~~~~~

Related to the storage work we are to improve the support for standardised data
formats like `csv`-files etc that can be loaded as recognised data from the
forward model.

Observation tooling
-------------------

In addition to the manual analysis allowed by the above mentioned data analysis
platform we are to implement tooling for automatic observation tooling. This is
particularly directed towards correlated observations and we expect to support
outlier detection, PCA scaling and localisation as part of the standard tooling
of ERT. In addition we want to make it possible for the users to write their
own elements that are to be part of the algorithmic pipeline in ERT.

First class support for sensitivity analysis
--------------------------------------------

Today the sensitivity analysis done in ERT is done via external jobs added to
the pipelines of ERT. Instead, we aim at making sensitivity analysis a first
class citizen in ERT that lets users sample from the parameter distributions
in a natural manner.

Server architecture
-------------------
As a first step towards a cloud agnostic solution that utilises its provided
resources in a good manner we plan to implement a server architecture. The
current recommendation is a three step solution:

Experiment server
~~~~~~~~~~~~~~~~~

A first step would be to spawn a server that deals with the running of a
particular experiment. This will disconnect the execution of an experiment from
the client and will allow the users with access to connect from different
machines to inspect the current status of the experiment.

Project server
~~~~~~~~~~~~~~

As a second solution we can spawn a server for the entire project. This would
make it possible to serve project data through an API, schedule runs on a
server that will be executed one after another etc. Since users will not take
down these servers themselves we will have to make them self terminating after
a certain amount of idle time.

ERT server
~~~~~~~~~~

Last, we can spawn an ERT-server that can serve information about the various
projects that are running. This would allow to keep the project servers running
by administrating them and serve connections via this single ERT server.
