External Workflows
==================

These jobs invoke an external program/script to do the job, this is
very similar to the jobs of the forward model, but instead of running
as separate jobs on the cluster - one for each realization, the
workflow jobs will be invoked on the workstation running ERT, and
typically go through all the realizations in one loop.

The executable invoked by the workflow job can be an executable you
have written yourself - in any language, or it can be an existing
Linux command like e.g. :code:`cp` or :code:`mv`.
