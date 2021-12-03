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
