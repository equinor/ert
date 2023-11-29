Concepts
========

Rationale
---------

This document is to explain key concepts in ERT. The motivation is not to come
up with a definition that is globally accepted, but to establish a domain
language for communication between ERT developers. In other words, explain the
ERT community's perspective on related concepts. Furthermore, the hope is that
the same terms can then be used for communication with related development
efforts and our users. With this in mind it is important to realise that the
text below is not the *ultimate truth* |TM|, but a basis for which we can
continue our discussions. This will be most valuable if we continuously compare
our codified data models, as well as our communication with users (through
dialog and user interfaces) towards these concepts. And it might very well be
that it is the below concepts that needs to be matured further.

.. |TM| unicode:: U+2122
    .. trademark sign

A second motivation for writing this is that as soon as ERT is exposing a
concept via its interface, that being through a configuration file, an API or a
graphical user interface there is at least one formal perspective of what this
concept is. To ensure a coherent user and developer experience it is beneficial
that different entities reasoning about a concept have a unified perspective.
It is not about being correct, it is about being relevant and coherent.

Ensembles and their evaluation
------------------------------

Forward model
~~~~~~~~~~~~~

The task of the forward model is to **deterministically** simulate the
behaviour of the model under study. In ERT a forward model is considered a
black box function. This yields the benefit of extremely flexible modelling at
the cost of low ability to reason about the forward model, its internal data
flow and to build tooling on top of it. Furthermore, an ERT forward model is
today implemented as a sequential list of scripts that are executed in the same
disk area such that data from one script can be dumped on disk and picked up by
the next job. More on the current and future data model and run environment of
the forward model can be found in the :doc:`forward_model`.

Template model
~~~~~~~~~~~~~~

ERT's domain is to facilitate modelling work under uncertainty. In particular,
multiple of the input variables to the forward model have an uncertainty
attached to them. We call these variables the *parameters* of the model and let
each parameter be represented by a probability distribution. The model we
obtain by combining the forward model and parameter distributions is referred
to as the *template model*. Notice that the template model is not a
deterministic model any more, but a model that tries to capture the uncertainty
of your current knowledge.

Realisation
~~~~~~~~~~~

The drawback of the template model is that depending on the forward model it
might be very difficult to simulate. We can however, by sampling all of the
parameters from their corresponding distributions, turn the template model into
a realisation. The realisation consists of the forward model :code:`fm`
together with its input variables :code:`x`. Observe that in a realisation
:code:`x` consists of only deterministic variables and is hence much easier to
simulate and reason about than the template model.

Ensemble
~~~~~~~~

As the name hints towards, ERT's approach to studying a template model, is to
create an ensemble of realisations. In particular, an ensemble is a list of
realisations. The idea is that with a sufficiently large ensemble the
uncertainty of the template model is represented by the ensemble.

Evaluating an ensemble
~~~~~~~~~~~~~~~~~~~~~~

Recall that a realisation consists of a forward model :code:`fm` together with input
:code:`x`. An observant reader might have realised that this, together with the
assumption that the forward model is deterministic, is sufficient to get hold
of the responses (output) of the forward model in this realisation. This
process of applying the forward model on the input to retrieve its response
:code:`y = fm(x)` is coined to evaluate a realisation. And similarly, if one
evaluates all of the realisations in an ensemble we say that we evaluate the
ensemble.  This separation between input data together with the method we want
to apply on it and the result might seem artificial at first. But the truth is
that in real life (as real as it gets on a computer at least) evaluations fail.
One can think of it as evaluating a realisation will with high probability give
you :code:`y = fm(x)`, but you might also get a failure (:code:`None`). Since
this is an inherent part of our domain we need to be able to communicate and
reason about it as a first class citizen of our problem domain.

Realisation response
~~~~~~~~~~~~~~~~~~~~

The response :code:`y = fm(x)` computed by successfully evaluating a realisation is
referred to as the realisation response. Since the forward model is assumed to
be deterministic the response of a realisation will be the same each time the
evaluation is successful.

Ensemble response
~~~~~~~~~~~~~~~~~

By evaluating an ensemble you will obtain an ensemble response. An ensemble
response contains the realisation responses for all of its realisations;
:code:`fm(x)` for all of the successful realisations and the failure state
:code:`None` for each of the failed realisations. By reevaluating an ensemble
multiple times one obtain multiple ensemble responses. Notice that the
realisation response of a successful realisation is always the same in the
different ensembles. Hence, one can merge two ensemble responses in a natural
way to obtain a merged ensemble response where all realisations that was
successful in at least one the two ensembles evaluations will be successful in
the merged ensemble response.
