.. _cha_workflow_jobs:

*************
Workflow jobs
*************

Workflow jobs are intended to run at specific points during optimization,
e.g. just before or after running a batch of simulations.

They are defined in the `install_workflow_jobs` section of the config file:

.. code-block:: yaml

  install_workflow_jobs:
    -
      name: <name used inside config>
      executable: <path to job executable>

Where the ``executable`` field may be an absolute path, or relative to the
location of the config file. The executable can be any program that can be
executed on the command line.

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
simulations. Each section contains a list of jobs, that will run in the order
they are specified.

Example
-------

Let's make a very simple script that just writes a file that contains the number
of batches that ran so far:

.. code-block:: python

  #!/usr/bin/env python
  import sys
  from pathlib import Path

  def main(filename):
      try:
          count = int(Path(filename).read_text())
      except FileNotFoundError:
          count = 0
      Path(filename).write_text(str(count + 1))

  if __name__ == "__main__":
      main(sys.argv[1])


Save it as ``count.py`` and make the script executable:

.. code-block:: bash

  chmod a+x count.py

Add it to the ``install_workflow_jobs`` section:

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
