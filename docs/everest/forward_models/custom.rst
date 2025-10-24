
.. _cha_creating_custom_jobs:


Custom workflow jobs
====================

To use a custom workflow job in everest, the job needs to be added to the `install_workflow_jobs` section of the config file.

The standard template to install a job inside the config file is as follows:

.. code-block:: yaml

    install_workflow_jobs:
      -
        name: <name used inside config>
        executable: <path to job executable>

Workflows can then be specified to run the installed jobs for specific triggers:

.. code-block:: yaml

    workflows:
      pre_simulation:
        - job_name <job arguments>
        - another_job <job arguments>
      post_simulation:
        - job_name <job arguments>
        - another_job <job arguments>

Currently `pre_simulation` and `post_simulation` triggers are defined, which run the specified jobs just before, and directly after, running each batch of simulations.

For each batch evaluation, a runpath file is written containing a list of the simulation folders. The location of that file can be passed to a job using the pre-defined `runpath_file` variable that will be replaced with the full path to the runpath file. For example, this will pass the location of the runpath file as the first argument to a workflow job:

.. code-block:: yaml

    workflows:
      pre_simulation:
        - job_name r{{ runpath_file }}
