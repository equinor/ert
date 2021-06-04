Development strategy
====================

For the goals of the ERT development itself we refer the reader to the
:ref:`Roadmap`. Here, we will instead give a direction for how the roadmap is
to be realised. Both as an overarching strategy, as well as tactical choices
and milestones along the way. None of this is set in stone and in particular
the tactical elements should be frequently discussed.

Strategy
--------

Cohesion and coupling
~~~~~~~~~~~~~~~~~~~~~
We aim at developing modularized software where each component have a single,
dedicated responsibility. The goal is that while implementation might be
complex, we should work hard to keep the responsibility of each component as
clear and concise as possible. Planning component purpose, documenting it and
carefully comparing interfaces with purpose should help us towards this. The
code of each component should be present solely to deliver on the purpose of
the component. Additionally, we should work hard to keep the coupling between
the components as low as possible. That is, their communication should be of
limited scope, passing data and events.

Strangulation
~~~~~~~~~~~~~
The strangler pattern is a well-known technique for iteratively transforming
legacy software. A very short and naive description is that you start by
identifying a responsibility in your software, then create a proxy in front of
the capability at hand that the rest of your system consumes. After which you
iterate between changing the implementation backing the proxy aiming at
isolating responsibilities and streamlining and splitting the proxy. For better
and more extensive explanations we refer the reader to the literature.

Evaluation, storage and analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For strangulation (and software development for that matter) to succeed, clear
and separate responsibilities are needed. The fundamental building blocks of
ERT are:

- evaluation: evaluate ensembles given the input data and the forward model to
  retrieve the model responses.
- storage: persist all data and state related to ERT experiments and expose the
  data via an API.
- analysis: given the input and output of the evaluation together with
  additional configuration the analysis module runs algorithms to conclude
  results or identify the next ensemble to be evaluated.

All of the three modules described above should be independent of each other.
That is, neither should dependent directly on another. Instead the business
logic that utilises the above capabilities should reside in the engine.

The ERT engine
~~~~~~~~~~~~~~
The evaluator, storage and analysis is joined in an experiment concept that
given a user configuration ensures that evaluation and analysis is invoked
in alternating fashion with the relevant results persisted in storage. The ERT
engine represents the business logic of ERT.

ert2 vs ert3
~~~~~~~~~~~~

Shared disk
"""""""""""
The user experience of ert2 is built upon the assumption that a shared disk
between the client and the compute nodes are available. Still, in large parts
of the codebase this assumption can be lifted or at the very least isolated.
ert3 will not assume the presence of such a disk.

Data model
""""""""""
ert3 is to be developed such that the data model of ert2 is a subset of the
ert3 data model.

Difference
""""""""""
Hence, the main difference in implementation between ert2 and ert3 should occur
in configuration, engine and user interface. In addition to the above
differences ert3 will introduce sensitivity analysis and optimization as
experiment types similarly to history matching and evaluation. Furthermore, a
workspace concept will be introduced to more naturally allow for multiple
experiments side by side.

Tactics
-------

This section contains a list of epics suggested to be carried out to continue
realizing the development strategy. It is expected that a section is
turned into an epic issue before it is launched.

Remove the legacy legacy evaluator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
An evaluation proxy was defined for which an implementation using the legacy
evaluation and an implementation based on prefect exists. However, ert2 only
consumes the legacy evaluator via the proxy if enabled by a feature flag. We
should make the proxy implementation production ready before making it the
default interaction, followed by deleting the possibility to interact with the
legacy evaluator without utilising the proxy.

Unify the data models of the new storage instances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Currently three implementations of ert-storage exists. The legacy storage
currently residing in libres, the version in the ert repository used for the
webviz instance and the version in the ert-storage repository used for ert3
testing. A first step to unite these implementations is to merge the
ert-storage implementations in ert and ert-storage such that the visualisation
instance and the ert3 implementation is backed by the same storage instance.

Make ERT mono-repo
~~~~~~~~~~~~~~~~~~
Due to fast moving modules and natural strangulation proxies that slice across
repositories it is viewed as beneficial to merge all ERT related repositories
into a single mono repository. That is, if the responsibility of a repository
cannot be explained without explaining ERT it should be moved into the mono
repository. We should start by moving libres and then afterwards plan for
moving ert-storage.

Single ert module
~~~~~~~~~~~~~~~~~
A single ert module with heavy usage of submodules. For the parts where ert2
and ert3 differs (engine, configuration and part of UI) we either introduce the
submodules ert.two and ert.three, or ert.engine2 and ert.engine. Either way,
shared code should be put in the ert3 module(s) such that down the road we can
remove ert2 entirely without touching ert3.

Logging
~~~~~~~
Currently there are numerous ways of logging in ert and libres. Furthermore,
the Kibana instance that used to be active on-premise was decommissioned. We
should get a new central logger up and running on-premise and make sure that
relevant loggig is forwarded to this central instance. This will allow us to
monitor usage of ERT and act upon it.

Implement a backend of the storage API using EnKFMain + file storage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To utilise strangulation techniques and to make the new visualisation solution
available to a larger set of users we intend to implement the storage API using
EnKFMain and file storage. It will rely heavily on shared disk access to user
configuration and will loose all data if the storage files are deleted from
disk. All in all, is should behave similarly to what EnKFMain and file storage
does today.

Make all data exposed to users pass through the storage API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
With an implementation of the storage API backed on EnKFMain and file storage
we are again ready to aim for all user facing data (visualisation and export)
to pass through the storage API.

Introduce blob records
~~~~~~~~~~~~~~~~~~~~~~
To pass blob data around in the evaluation in ert3 we need to expose the
possibility to internalise and pass around blob data. It should culminate in
the SPE1 example no longer having to depend on `cp` to move the datafile to the
compute node.

Drop the Qt-plotter
~~~~~~~~~~~~~~~~~~~
With the webviz backed plotter available to all users we should drop the Qt
based plotter.

Increase visual scalability of the new visualiser
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The product owner has a list of improvements to make the visualiser scale
better visually for large cases. We should gather these into a milestone of
issues.

Configurable compute environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The compute environment used, in particular for unix steps, should be
configurable in a natural manner. It should be possible to configure an
extension of a komodo environment on-premise. As in, additional Python packages
and single-file scripts should be possible to configure. This should also be
possible to do in a setup without Komodo.

Plugged-in sensitivity
~~~~~~~~~~~~~~~~~~~~~~
As a first example of an open core interaction we should make the current
sensitivity algorithm pass its options to ert3 via a json schema (or a similar
technology), for which ert3 then makes those options available in the configuration
and passes the configured values back to the algorithm when the experiment
launches. The goal is to have a loose coupling to the extent that the algorithm could be
proprietary without violating the GPLv3 license of ERT.

Implement optimization concepts in ERT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Implement an as simple as possible optimisation algorithm, together with an
introduction of optimization to the configuration in ert3, necessary business
logic in the engine and the capability to store control variables in storage.

Pilot ready storage backend in Azure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Before ERT3 can run in the cloud we need to implement and set up deploy of a
pilot ready storage solution in Azure.

Pilot ready new evaluator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Before ERT3 can run in the cloud we need to prepare the new ensemble evaluator
for pilot in Azure.

Temporary storage for evaluator and analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To start decreasing the coupling between the evaluator and EnKFMain we start by
making the evaluator write to its separate version of file storage. Afterwards
it makes the data available for EnKFMain such that it can be persisted in file
storage. The goal is to decouple the storage mechanism of the evaluator and
EnKFMain. A similar approach should be taken for the analysis module.

Make a proxy for the analysis module to facilitate strangulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Write a proxy on top of the analysis module together with extensive tests as a
starting point for strangulation.

Support history matching in ERT3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Using the above implemented analysis proxy we are to implement history matching
capabilities in ert3.

Introduce ensemble and analysis workflows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Implement support for ensemble and analysis workflows that can be executed both
in ERT2 and ERT3. In ERT2 this will be introduced as new hooks, while in ERT3
we are to implement a pipeline system (probably based on the same workflow
manager as used for the forward model) and use it.

Experiment server
~~~~~~~~~~~~~~~~~
Move the logic of the ERT engine into a server for which the clients can
interact with.
