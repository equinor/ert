Development strategy
====================

Here we describe the basic strategies how ert achieves general goals.
Both as an overarching strategy, as well as tactical choices
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
