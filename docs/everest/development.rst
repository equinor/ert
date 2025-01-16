.. _cha_development:

***********
Development
***********

In this section Everest development decisions are documented.


Architecture
============

The everest application is split into two components, a server component and a
client component.

.. figure:: images/architecture_design.png
    :align: center
    :width: 700px
    :alt: Everest architecture

    Everest architecture

Every time an optimization instance is ran by a user, the client component of the
application spawns an instance of the server component, which is started either on a
cluster node using LSF (when the `queue_system` is defined to be *lsf*) or on the
client's machine (when the `queue_system` is defined to be *local*).

Communication between the two components is done via an HTTP API.


Server HTTP API
===============
The Everest server component supports the following HTTP requests API. The Everest
server component was designed as an internal component that will be available as
long as the optimization process is running.


.. list-table:: Server HTTP API
   :widths: 25 25 75
   :header-rows: 1

   * - Method
     - Endpoint
     - Description
   * - GET
     - '/'
     - Check server is online
   * - GET
     - '/sim_progress'
     - Simulation progress information
   * - GET
     - '/opt_progress'
     - Optimization progress information
   * - POST
     - '/stop'
     - Signal everest optimization run termination. It will be called by the client when the optimization needs to be terminated in the middle of the run


Everest vs. Ert data models
===========================
Everest uses Ert for running an experiment, but instead of submitting an `ensemble` (i.e., as in Ert) to the queue we submit
a `batch` in Everest. `Batches` are in principle very similar to `ensembles`, but they have some key differences.
The purpose of this section is to explain these key differences from a developer point-of-view.

In Ert, an `ensemble` contains a number of `realizations` which are a set of `model parameters` which Ert attempts to history match to some data.
These `model parameters` are sampled from a certain distribution after creating the `ensemble`. In Everest we are not history matching our `model parameters`,
but instead try to find an optimal strategy on how to operate the particular model(s) in order to maximize some objective (i.e., we optimize for a set of `controls`).
Simply put, the optimization algorithm is iteratively updating our controls until we reach some convergence criteria (i.e., have obtained the optimal strategy).

In order to perform the optimization we need to get the sensitivity of the `objective function` to these `controls` or `optimization variables`.
It is important to understand how are current controls are performing. Did we improve our strategy and therefore our objective function value?
Note: `updated controls` after one optimization iteration become `current controls` for the next iteration. This means in Everest there will be a
distinction between running a forward model for `current controls` or `perturbed controls`. The forward model doesn’t care for which type of controls
it is currently running, but the optimizer will handle the results slightly different.

If we perform robust optimization (i.e., don’t have a single deterministic underlying model) a `batch` contains a certain number of `realizations`
(similar to `realizations` in Ert except static and denoted with `<GEO_ID>`) and each `realization` contains a certain number of `simulations`
(i.e., forward model runs). The `simulations` are either evaluating the objective function value for `current controls` and/or for
each `perturbed controls`. This means that the hierarchy of the output in Everest and Ert are different (Fig 2).

.. figure:: images/Everest_vs_Ert_01.png
    :align: center
    :width: 700px
    :alt: Everest vs. Ert data models

    Difference between `ensemble` in Ert and `batch` in Everest.
    A `realization` in Everest refers to a static model configuration which doesn’t change during the optimization,
    but is different from model to model. While `realization` in Ert means set of model parameters which are going to be history
    matched to certain data. Particularly, in Ert the `model parameters` or `realizations` are actually the objective of the optimization,
    while in Everest the optimization objective is finding the correct controls such that a certain objective is maximized or minimized.
    Since Everest still uses Ert to submit the `batch` (i.e., `ensemble`) to the queue, the Everest runs (i.e., for a <GEO_ID> we run a set of controls)
    are mapped to Ert `realizations` accordingly. After collecting the results for each Ert `realization` they are mapped back to Everest structure.

The mapping from data models in Everest to Ert is the same as flatten a 2D array (i.e., from index based on `<GEO_ID>` and `simulation` in Everest to
index based on `realization` Ert). Ultimately,  Ert is submitting the forward model runs to the queue and is agnostic about the meaning of each run.
Only when the data is collected back in Everest is the meaning of each run attributed.

In Ert each `ensemble` is exactly one step in the history matching algorithm and `realizations` have continuity from one iteration to the next.
For example, `model parameter set 0` are smoothly(?) changing from prior to posterior over the course of the history matching.
This is not the case for `simluations` in Everest and highlights another key difference. A `batch` can contain several different configurations (Fig 3)
and `simulation 0` for `<GEO_ID> = 0` can be either `current` or a `perturbed control` hence there is no continuity from one `batch` to the next. `<GEO_ID>`
is continuous from one `batch` to the next since they are not changing at all over the course of the optimization.

.. figure:: images/Everest_vs_Ert_02.png
    :align: center
    :width: 700px
    :alt: Other `batch` configurations Everest

    Two other possible configurations of Everest `batches` in the context of gradient-based optimization algorithms (i.e., `optpp_q_newton`).
    Option (A) can occur when the option `speculative` is set to False, this means that before proceeding with the optimization the update set of controls is
    first evaluated if it actually improves the objective function value (hence the `batch` contains only `current controls` for each forward model run for each
    `<GEO_ID>`). Option (B) can occur when the `current controls` are already evaluated in a previous `batch` and no update to current controls has occurred yet.
    In the context of gradient-free optimization methods a `batch` can also contain multiple current control forward model runs (ask Pieter how batches change
    depending on the optimization algorithm, I wrote this now based on his old figure, but pretty sure the terminology is wrong here).

Perhaps an attempt could be made to improve the terminology on the Everest side, at least from a developer point of view. `Simulation`
is meaningful to the user and perhaps should be kept as is, but `simulation` to a Everest developer is not helpful since in the code `realization`
is mapped to `simulation` and vice versa. Also, `<GEO_ID>` is misleading in several ways, it’s not the same as a `realization` in Ert since it’s static
and furthermore we could be optimizing anything which has nothing to do with “GEO”. Perhaps we could change `<GEO_ID>` to `<STATIC_MODEL_ID>`
to emphasize the fact these model parameters are not changing over the course of the optimization. And for developers it would be much clearer to
change `simulation` to `ert_realization` or `forward_model_evaluation`.
