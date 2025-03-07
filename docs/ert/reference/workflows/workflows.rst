External workflow jobs
======================

External workflow jobs invoke programs and scripts that are not bundled with ERT,
which makes them similar to jobs defined as part of forward models.
The difference lies in the way they are run.
While workflow jobs are run on the workstation running ERT
and go through all the realizations in one loop, forward model jobs run in parallel on HPC clusters.

The executable invoked by the workflow job can be an executable you
have written yourself - in any language, or it can be an existing
Linux command like e.g. :code:`cp` or :code:`mv`.

Internal workflow jobs
======================

.. warning::
    Internal workflow jobs are under development and the API is subject to changes

Internal workflow jobs is a way to call custom python scripts as workflows. In order
to use this, create a class which inherits from `ErtScript`:

.. code-block:: python

   from ert import ErtScript

   class MyJob(ErtScript):
       def run(self):
           print("Hello World")

ERT will initialize this class and call the `run` function when the workflow is called,
either through hooks, or through the gui/cli.

The `run` function can be called with a number of arguments, depending on the context the workflow
is called. There are three distinct ways to call the `run` function:

1. If the `run` function is using `*args` in the `run` function, only the arguments from the user
configuration is passed to the workflow:

.. code-block:: python

   class MyJob(ErtScript):
       def run(self, *args):
           print(f"Provided user arguments: {args}")

.. note::
    The name of the argument is not required to be `args`, that is just convention.

2. There are a number of specially named arguments the user can call which gives access to internal
state of the experiment that is running:


.. autoclass:: ert.config.workflow_fixtures.WorkflowFixtures
    :members:
    :undoc-members:
    :exclude-members: parent, random_seed, reports_dir, observation_settings, es_setting, run_paths


.. note::
    The current ensemble will depend on the context. For hooked workflows the ensemble will be:
    `PRE_SIMULATION`: parameters and no reponses in ensemble
    `POST_SIMULATION`: parameters and responses in ensemble
    `PRE_FIRST_UPDATE`/`PRE_UPDATE`: parameters and responses in ensemble
    `POST_UPDATE`: parameters and responses in ensemble
    The ensemble will switch after the `POST_UPDATE` hook, and will move from prior -> posterior

.. code-block:: python


    class MyJob(ErtScript):
        def run(
            self,
            workflow_args: List,
            ert_config: ErtConfig,
            ensemble: Ensemble,
            storage: Storage,
        ):
            print(f"Provided user arguments: {workflow_args}")

For how to load internal workflow jobs into ERT, see: :ref:`installing workflows <legacy_ert_workflow_jobs>`
