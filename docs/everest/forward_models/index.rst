
.. _cha_forward_model_jobs:


******************
Forward model jobs
******************

.. toctree::
    :hidden:

    builtin
    custom

During optimization the values of the objective functions and output constraints
are calculated by a series of jobs specified in the `forward_model` section of
the configuration file:

.. code:: yaml

  forward_model:
    - job-name <arguments>
    - job-name <arguments>

These jobs are executed in the order they are specified, and exchange data via
files. Before starting a new batch of forward model jobs, EVEREST writes an
input file containing the current control values and copies other user-specified
input files to the location in the file system where the forward model jobs will
run. Jobs can produce output files that serve as inputs for jobs that are
started later. Finally, the outputs of the forward model (objective and
constraint values) are saved, read by EVEREST and send back to the optimizer.

Forward model jobs are command line applications that are run by the EVEREST job
scheduler. A few built-in jobs are provided, which are described in the section
:ref:`cha_builtin_forward_model_jobs`. The EVEREST installation is generally
providing additional forward models via its plugin system, such as those
described in :ref:`cha_everest_models`. The user can also add their own forward
models using the mechanism described in :ref:`cha_creating_custom_jobs`.
