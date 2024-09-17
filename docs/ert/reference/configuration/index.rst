Configuration
=============

ERT requires some set-up. For example:

* Distributions must be assigned to priors.
* The forward model must be described so that ERT can run it.
* Observations must be provided.
* The analysis algorithms you want to use must be parameterized.
* If running on an HPC cluster, you will have to tell ERT how to interact
  with the queue system.

There are prescribed ways to provide all of this information, and ERT has
strict formatting requirements for its configuration file. However, it also
provides convenient syntax and structure for dealing with almost any scenario.

This chapter sets out everything you need to know to configure ERT.

.. toctree::
   :maxdepth: 2

   data_types
   forward_model
   observations
   keywords
   queue
   magic_strings
   site_wide
