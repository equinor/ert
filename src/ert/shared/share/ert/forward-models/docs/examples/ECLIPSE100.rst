The version, number of cpu, and whether or not to ignore errors and whether or not to produce `YOUR_CASE_NAME.h5` output files can
be configured in the configuration file when adding the job, as such:

.. code-block:: bash

    FORWARD_MODEL ECLIPSE100(<ECLBASE>, <VERSION>=xxxx, <OPTS>={"--ignore-errors", "--summary-conversion"})

The :code:`OPTS` argument is optional and can be removed, fully or partially.
In absence of :code:`--ignore-errors` eclipse will fail on errors.
Adding the flag :code:`--ignore-errors` will result in eclipse ignoring errors.

And in absence of :code:`--summary-conversions` eclipse will run without producing `YOUR_CASE_NAME.h5` output files.
Add flag :code:`--summary-conversions` to produce `YOUR_CASE_NAME.h5` output files.
