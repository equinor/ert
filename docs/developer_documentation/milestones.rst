Milestones
==========

This section contains a list of milestones suggested to be carried out to
continue realizing the development strategy. It is expected that a section is
turned into an epic issue before it is launched.

Make all data exposed to users pass through the storage API [in progress]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
With an implementation of the storage API backed on EnKFMain and file storage
we are again ready to aim for all user facing data (visualisation and export)
to pass through the storage API. Ensure that this can be done while ERT is
running. Deprecate the collectors so that we can make them private later.

Move the responsibility of fetching and storing data out of the analysis module [in progress]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
As a first step towards further isolating the analysis pipeline of ERT and
opening up for the possiblity of a stateless Python API for analysis we are to
separate the responsibility of data fetching and storing (including the
knowledge of EnKFMain and storage) and all logic for how analysis is done. In
addition, we should improve the test base and seek local code improvements.

Data record maturation [in progress]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Extend the functionality of the records to be able to carry blob data,
hierarchical data (including summary data), support transformations and
serialization of record data.

Act upon feedback on new data visualiser
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Actively seek feedback on webviz-ert and act upon the feedback such that
webviz-ert can replace the Qt-plotter as the standard visualisation tool.

Drop the Qt-plotter
~~~~~~~~~~~~~~~~~~~
With the webviz backed plotter available to all users we should drop the Qt
based plotter.

ERT-storage discrepancies
~~~~~~~~~~~~~~~~~~~~~~~~~
Currently, there is a difference in how ERT2 and ERT3 stores data in
ERT-storage. They should unify such that webviz-ert can visualise data from
both tools. Furthermore, we should abandon all usage of `userdata` from both
ERT2 and ERT3.

Delete the deprecated interaction with the queue system
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
After production setting the ensemble evaluator, we should phase out the
possibility of interacting with the queue system bypassing the legacy
evaluator.

ERT3 monitoring
~~~~~~~~~~~~~~~
Make it possible to monitor the progress of the realisations when running
experiments in ERT3. The solution can be cli- or gui-based, but it should be
shared between ERT2 and ERT3.

Introduce an experiment concept
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To improve the synergies between ERT2 and ERT3 a shared implementation of an
experiment should be introduced - with the responsibility of executing a single
experiment. This implementation should contain the logic of the current run
models in ERT2 and parts of the engine logic in ERT3.

Handle failing realisations in ERT3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ERT3 needs to handle failing realizations, both when running and persisting the
information.

Configurable compute environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The compute environment used, in particular for unix steps, should be
configurable in a natural manner. It should be possible to configure an
extension of a komodo environment on-premise. As in, additional Python packages
and single-file scripts should be possible to configure. This should also be
possible to do in a setup without Komodo.

Pilot ready storage backend in Azure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Before ERT3 can run in the cloud we need to implement and set up deploy of a
pilot ready storage solution in Azure.

Pilot ready new evaluator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Before ERT3 can run in the cloud we need to prepare the new ensemble evaluator
for pilot in Azure.

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

Implement Everest-based plugin for optimization and sensitivity analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
With support for plugin optimization and sensitivity algorithms we should
provide a plugin for each based on the current Everest algorithm.

Revise FMU-FAQ
~~~~~~~~~~~~~~
The FMU FAQ currently contains some issues or frequently asked questions that
could be solved via implementation in ERT. We should collect those and act upon
them.

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
