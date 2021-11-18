The version, number of cpu and whether or not to ignore errors can
be configured in the configuration file when adding the job, as such:

.. code-block:: bash

    FORWARD_MODEL ECLIPSE300(<ECLBASE>, <VERSION>=xxxx, <OPTS>="--ignore-errors")

The :code:`OPTS` argument is optional and can be removed, thus running eclipse
without ignoring errors
