External workflow jobs
======================

External workflow jobs invoke programs and scripts that are not bundled with ERT, 
which makes them similar to jobs defined as part of forward models.
The difference lies in the way they are run.
While workflow jobs are run on the workstation running ERT
and go through all the realizations in one loop, forward model jobs run in parallell on HPC clusters. 

The executable invoked by the workflow job can be an executable you
have written yourself - in any language, or it can be an existing
Linux command like e.g. :code:`cp` or :code:`mv`.
