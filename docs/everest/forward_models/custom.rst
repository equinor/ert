.. _cha_creating_custom_jobs:

*************************
Custom forward model jobs
*************************

To use a custom job in everest, it needs to be added to the ``install_jobs``
section of the config file:

.. code-block:: yaml

    install_jobs:
    -
        name: <name used inside config>
        executable: <path to job executable>

Where the ``executable`` field may be an absolute path, or relative to the
location of the config file. The executable can be any program that can be
executed on the command line.

Here we show an example of implementing a simple custom forward job in Python.


Example: prioritizing a group of wells
--------------------------------------

Our example job will part of a forward model that consists of several jobs that
run in sequence. These jobs generally communicate by reading and writing to
files: the output of one job may serve as the input for the next job. We will
create a job that accepts a file containing well priorities and outputs a file
with re-arranged priorities.

We assume that some previous job created a file ``priorities.json``, which will
be the input for our new job:

.. code-block:: json

  {
      "PROD2": 9,
      "PROD1": 10,
      "INJ1": 8,
      "INJ2": 7,
      "PROD3": 6,
      "INJ3": 5,
      "PROD4": 4,
      "PROD5": 3,
      "PROD6": 2,
      "PROD7": 1
  }

We would like our job to re-assign priorities, such that "PROD1", "PROD4" and
"INJ3" are assigned higher priorities than all other wells, while retaining
their relative order.


Implementing the Python script
------------------------------

The input file contains a group of wells that we would like to be drilled first,
in order of their priority. Then all the other wells will be added afterwards,
again in the order of their priority.

To implement this, we first write a Python function that splits the wells into
two groups, sorts each group again and re-prioritizes them:

.. code-block:: python

  def rearrange_priorities(well_priorities, first_wells):
      # Find the wells that should be drilled first and sort them by priority:
      first = sorted(
          [well_name for well_name in well_priorities if well_name in first_wells],
          key=lambda x: well_priorities[x],
          reverse=True,
      )
      # Find the rest of the wells and sort them by priority:
      rest = sorted(
          [well_name for well_name in well_priorities if well_name not in first_wells],
          key=lambda x: well_priorities[x],
          reverse=True,
      )

      # Make a list of wells in the new order:
      new_order = first + rest

      # Assemble the new dict from the ordered list with new priorities:
      new_priorities = dict(zip(new_order, range(len(new_order), 0, -1)))

      return new_priorities


The input priorities are likely the result from another job that runs before our
new job, and the output file is likely going to be used by the next job.
Therefore we assume that these are in JSON format, which is commonly used to
exchange data that should be machine-readable.

We assume that the user provides the file listing the wells that we want to
prioritize. For this we use the YAML format, which is more human-friendly.

We create a main function that reads the original priorities, and the requested
wells from the input files, and writes the result to an output file. We use the
Python ``argparse`` module to define command line arguments that will be used to
set the file names:


.. code-block:: python

  def main(argv):
      arg_parser = argparse.ArgumentParser()
      arg_parser.add_argument("--input", type=str)
      arg_parser.add_argument("--output", type=str)
      arg_parser.add_argument("--prioritize", type=str)
      options, _ = arg_parser.parse_known_args(args=argv)

      with open(options.input, encoding="utf-8") as fp:
          well_priorities = json.load(fp)

      with open(options.prioritize, encoding="utf-8") as fp:
          first_wells = yaml.safe_load(fp)

      new_priorities = rearrange_priorities(well_priorities, first_wells)

      with open(options.output, "w") as fp:
          json.dump(new_priorities, fp, sort_keys=False, indent=2)


We can put this all together in a script ``prioritize.py``:

.. code-block:: python

  import json
  import sys
  import argparse
  import yaml


  def rearrange_priorities(well_priorities, first_wells):
    ...  # replace with the function defined above


  def main(argv):
    ...  # replace with the main function defined above


  if __name__ == "__main__":
      main(sys.argv[1:])



Testing the python script:
--------------------------

To test the script we write the user input file ``wells.yaml``:

.. code-block:: yaml

  - PROD4
  - PROD1
  - INJ3

And run the script:

.. code-block:: bash

  python prioritize.py --input priorities.json --prioritize wells.yaml --output sorted.json



Inspecting the ``sorted.json`` file shows that the requested wells are indeed
prioritized first, in the correct order:

.. code-block:: json

  {
    "PROD1": 10,
    "INJ3": 9,
    "PROD4": 8,
    "PROD2": 7,
    "INJ1": 6,
    "INJ2": 5,
    "PROD3": 4,
    "PROD5": 3,
    "PROD6": 2,
    "PROD7": 1
  }


Deploying the new script:
-------------------------

To make our new script useful in a Everest case, it needs to be added to the
``install_jobs`` section of the configuration file. This requires the location
where we saved the script, and it requires the script to be executable. This is
most conveniently done by adding a so-called "shebang" line at the beginning of
the script:

.. code-block:: python

  #!/usr/bin/env python

  ... # Insert the rest of the script

and making the script executable:

.. code-block:: bash

  chmod +x prioritize.py

Add it to the ``install_data`` section:

.. code-block:: yaml

  install_jobs:
  -
      name: prioritize
      executable: jobs/prioritize.py

and it can be used as a job in your forward model using the command name
``prioritize``.
