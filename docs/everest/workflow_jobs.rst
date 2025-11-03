.. _cha_workflow_jobs:

*************
Workflow jobs
*************

Workflow  jobs are intended to run at specific points during optimization,
e.g. just before or after running a batch of simulations.

They are defined in the `install_workflow_jobs` section of the config file:

.. code-block:: yaml

  install_workflow_jobs:
    -
      name: <name used inside config>
      executable: <path to job executable>

This is very similar to the way forward model jobs are installed, refer to
:ref:`cha_creating_custom_jobs` for more details.

The `workflows` section is used to specify what triggers the workflow jobs to
run:

.. code-block:: yaml

  workflows:
    pre_simulation:
      - job_name <job arguments>
      - another_job <job arguments>
    post_simulation:
      - job_name <job arguments>
      - another_job <job arguments>

Currently `pre_simulation` and `post_simulation` triggers are defined, which run
the specified jobs just before, and directly after, running each batch of
simulations.

Example
-------

Let's make a very simple script that just writes a file that contains the number
of batches that ran so far:

.. code-block:: python

  #!/usr/bin/env python
  import sys

  def main(filename):
      try:
          # Get the batch count from the file:
          with open(filename, "r", encoding="utf-8") as f:
              count = int(f.read().strip())
      except FileNotFoundError:
          # If the file does not exists, the count is zero:
          count = 0

      # Increase the count, and write it back:
      with open(filename, "w", encoding="utf-8") as f:
          f.write(f"{int(count + 1)}\n")

  if __name__ == "__main__":
      main(sys.argv[1])


Save it as ``count.py`` and make the script executable:

.. code-block:: bash

  chmod a+x count.py

Add it to the ``install_jobs`` section:

.. code-block:: yaml

  install_workflow_jobs:
  -
      name: count
      executable: count.py

Add it as a workflow job that should run after each batch of simulations:

.. code-block:: yaml

  workflows:
    post_simulation:
      - count r{{configpath}}/count.txt

After running the optimization you should find a file ``count.txt`` that
contains the number of batches that have run.

.. tip::
  For each batch evaluation, a runpath file is written containing a list of the
  simulation folders. The location of that file can be passed to a job using the
  pre-defined `runpath_file` variable that will be replaced with the full path to
  the runpath file (see :ref:`config_variables` for details on variables).

  For example, this will pass the location of the runpath file as the first argument
  to a workflow job:

  .. code-block:: yaml

    workflows:
      pre_simulation:
        - job_name r{{runpath_file}}/count.txt
