Glossary
========

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

.. glossary::

    forward model
        The task of the forward model is to **deterministically** simulate the
        behaviour of the model under study. In ERT a forward model is considered a
        black box function. This yields the benefit of extremely flexible modelling at
        the cost of low ability to reason about the forward model, its internal data
        flow and to build tooling on top of it. Furthermore, an ERT forward model is
        today implemented as a sequential list of scripts that are executed in the same
        disk area such that data from one script can be dumped on disk and picked up by
        the next step. More on the current and future data model and run environment of
        the forward model can be found in the :doc:`configuration/forward_model`.

    realisation
        The drawback of the template model is that depending on the forward model it
        might be very difficult to simulate. We can however, by sampling all of the
        parameters from their corresponding distributions, turn the template model into
        a realisation. The realisation consists of the forward model :code:`fm`
        together with its input variables :code:`x`. Observe that in a realisation
        :code:`x` consists of only deterministic variables and is hence much easier to
        simulate and reason about than the template model.

    ensemble
        As the name hints towards, ERT's approach to studying a template model, is to
        create an ensemble of realisations. In particular, an ensemble is a list of
        realisations. The idea is that with a sufficiently large ensemble the
        uncertainty of the template model is represented by the ensemble.

    experiment
        An experiment consists of one or more ensembles, which may be related via
        zero or more update steps. For example you might have a prior ensemble that
        results in the posterior ensemble by running IES.

    evaluating an ensemble
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

    realisation response
        The response :code:`y = fm(x)` computed by successfully evaluating a realisation is
        referred to as the realisation response. Since the forward model is assumed to
        be deterministic the response of a realisation will be the same each time the
        evaluation is successful.

    ensemble response
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

    simulation
        There are two uses of the word simulation:
         * A synonym for running a realization, ie. executing all the forward model steps.
         * Some forward model steps are simulators which do simulation, eg. OPM flow and eclipse.

    reservoir simulator
        Simulation of reservoir fields come in many forms, but for the purposes of
        ert we only consider simulators that produces
        :term:`summary files` as output. This includes, OPM flow and Eclipse.

    summary files
        The result of running a reservoir simulator is a number of time vectors
        which are written to summary files. See `OPM Flow manual`_ section F for details.

    summary key
        A summary key is a colon separated list of the required properties
        needed to uniquely specify a summary vector. What properties are
        required is specified in `OPM Flow manual`_ section
        F.9.2. Summary variables are described in the `OPM Flow manual`_
        section 11.1.
        A summary vector is uniquely specified by giving a summary variable, and
        potentially one or more of the following properties: well name, region name, lgr
        name, block index, completion index, network name.

    Repeat Formation Tester
    RFT
        RFT is short for Repeat Formation Tester which is a wireline formation
        tester that measures formation pressure. However, in this context RFT
        usually means measurements or simulated data for formation pressure and
        saturation.

    True Vertical Depth
    TVD
        The depth (in the vertical plane) of a point along a wells borehole.

    Measured Depth
    MD
        The length (along the well path) for a point along a well borehole.

.. _OPM Flow manual: https://opm-project.org/wp-content/uploads/2023/06/OPM_Flow_Reference_Manual_2023-04_Rev-0_Reduced.pdf
