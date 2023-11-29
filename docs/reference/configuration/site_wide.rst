Site wide configuration
=======================

When the ert application starts it will first read a *site configuration* file
before loading the configuration file given by the user. The path to the
configuration file is *compiled in* when building the `libres` library,
alternatively it can be configured runtime by setting the environment variable
`ERT_SITE_CONFIG`.

The purpose of the site-config file is to configure settings which should apply
to all users in an organisation, typical things to include in the site config
file is installation of forward model and workflow jobs, and configuring
properties of the cluster. This could be an example site config file: ::

   -- Set some properties of the local LSF system
   QUEUE_OPTION LSF LSF_QUEUE hmqueue
   QUEUE_OPTION LSF MAX_RUNNING 100
   QUEUE_OPTION LSF LSF_SERVER   lsf-front01.company.com
   QUEUE_OPTION LSF BSUB_CMD     /path/lsf/bin/bsub
   QUEUE_OPTION LSF BJOBS_CMD    /path/lsf/bin/bjobs
   QUEUE_OPTION LSF BKILL_CMD    /path/lsf/bin/bkill


   -- The location of the forward model and workflow jobs
   -- distributed with the libres source code. The part
   -- below here is just copied in from the libres source:

   WORKFLOW_JOB_DIRECTORY workflows/jobs/internal-gui/config

   JOB_SCRIPT ../../bin/job_dispatch.py
   INSTALL_JOB_DIRECTORY forward-models/res
   INSTALL_JOB_DIRECTORY forward-models/shell
