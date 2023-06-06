Workflows
=========

The Forward Model in ERT runs in the context of a single realization,
i.e. there is no communication between the different processes, and
the jobs are run outside of the main ERT process.

As an alternative to the forward model, ERT has a system with
*workflows*. Workflows allow you to automate combersome ERT processes,
as well as invoke external programs. The workflows are
run serially on the workstation actually running ERT, and should not
be used for computationally heavy tasks.

Configuring workflows in ERT consists of two steps: *installing the
jobs* which should be available for ERT to use in workflows, and then
subsequently assemble one or more jobs, with arguments, in a
workflow. You can use predefined workflow jobs, or create your
own. There are no predefined complete workflows.

The workflow jobs are quite similar to the jobs in the forward model,
in particular the jobs are described by a configuration file which
resembles the one used by the forward model jobs. The workflow jobs
can be of two fundamentally different types - *external* and *internal*.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Workflows

   external
   configuring_jobs
   complete_workflows
   added_workflow_jobs
