.. _cha_creating_custom_jobs:

*************************
Custom forward model jobs
*************************

To use a custom job in everest, the job needs to be added to the `install_jobs` section of the config file.

The standard template to install a job inside the config file is as follows:

.. code-block:: yaml

    install_jobs:
    -
        name: <name used inside config>
        source: <path to job config file>

Where the job config file is a file calls a python file. These scripts generally only contain one line:

.. code-block:: bash

    EXECUTABLE <python_file_path>.py

This file should point to the location of the python file relative to the job config file.
In this case the python file is next to the script.

--------
Examples
--------

This section outlines a few examples of jobs you might want to implement.
These examples are focussed on manipulating the optimizers output values concerning well priority.

All examples assume you have the following python dictionary in memory:

.. code-block:: python

  well_priorities = {
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


A group of wells first
----------------------

In this example, we have a group of wells that we would like to be drilled first, in order of their priority.
Then all the other wells will be added afterwards, again in the order of their priority.

Given an extra list of wells that should be the highest priority,
an implementation of this behaviour could be:

.. code-block:: python

  def rearrange_priorities(well_priorities, first_wells):
      first = sorted(
          [well_name for well_name in well_priorities if well_name in first_wells],
          key=lambda x: well_priorities[x],
          reverse=True,
      )
      rest = sorted(
          [well_name for well_name in well_priorities if well_name not in first_wells],
          key=lambda x: well_priorities[x],
          reverse=True,
      )

      new_order = first + rest
      sorted_priorities = sorted(well_priorities.values(), reverse=True)
      new_priorities = dict(zip(new_order, sorted_priorities))

      return new_priorities

In the above code block, we first split the wells in ``wells_priority`` based on whether they are in the ``first_wells`` list.
Then we order those lists and put the ``first`` before the ``rest``.
Then we assign ``new_priorities`` by using previous priority values based on the new order of wells.

We can then store these new priorities in an ``output_filename``
and make the python file execute the code when called as follows:

.. code-block:: python

  import yaml

  def entry_point():
      well_priorities = {
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
      first_wells = ["PROD4", "PROD1", "INJ3"]

      new_priorities = rearrange_priorities(well_priorities, first_wells)

      with open(output_filename, "w") as f:
          yaml.dump(new_priorities, f, sort_keys=False)

  if __name__ == "__main__":
      entry_point()

``output_filename``:

.. code-block:: yaml

  PROD1: 10
  INJ3: 9
  PROD4: 8
  PROD2: 7
  INJ1: 6
  INJ2: 5
  PROD3: 4
  PROD5: 3
  PROD6: 2
  PROD7: 1

where you can see the group has been correctly shifted to the front.

This can be expanded by loading both ``well_priorities`` and ``first_wells`` from files or from input arguments of the job.

Highest priority in Nth spot
-----------------------------

In this example, we want to always put the highest priority well from a group of wells on spot ``N`` in the order.
So the highest priority well could be pushed down to spot ``N``.
Note that ``N`` will be zero-indexed in this example, so if we want to give a well the top spot, ``N = 0``.

Given a priority_number and a list of candidates:

.. code-block:: python

  def shift_well(well_priorities, candidates, prio_num):
      sorted_priorities = sorted(well_priorities.values(), reverse=True)
      old_order = sorted(well_priorities.keys(), reverse=True, key=lambda w: well_priorities[w])

      best_candidate = max(candidates, key=lambda c: well_priorities[c])
      old_order.remove(best_candidate)

      new_order = old_order[:prio_num] + [best_candidate] + old_order[prio_num:]
      new_priorities = dict(zip(new_order, sorted_priorities))

      return new_priorities

Where we first determine the ``best_candidate`` by picking the highest priority one.
Then we remove that one from the original order and insert it in spot number ``prio_num``.
Then we assign ``new_priorities`` by using previous priority values based on the new order of wells.

We can then store these new priorities in an ``output_filename``
and make the python file execute the code when called as follows:

.. code-block:: python

  import yaml

  def entry_point():
      well_priorities = {
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
      candidates = ["PROD2", "PROD4", "INJ3"]
      prio_num = 4

      new_priorities = shift_well(well_priorities, candidates, prio_num)

      with open(output_filename, "w") as f:
          yaml.dump(new_priorities, f, sort_keys=False)


  if __name__ == "__main__":
      entry_point()

``output_filename``:

.. code-block:: yaml

  PROD1: 10
  INJ1: 9
  INJ2: 8
  PROD3: 7
  PROD2: 6
  INJ3: 5
  PROD4: 4
  PROD5: 3
  PROD6: 2
  PROD7: 1

Where you can see ``PROD2`` has been successfully shifted.


Well cycles
-----------

In this example, we want to repeat a specific cycle of wells:
2 producers, then one injector. This is done by splitting injectors and producers into groups.

In order to make ``well_priorities`` adhere to the cycle as well as possible,
we can implement the functionality as follows:

.. code-block:: python

  from itertools import cycle, islice

  def apply_cycle(well_priorities, config):
      well_cycle = islice(cycle(config["cycle"]), len(well_priorities))
      groups = {k: sorted(v, key=lambda x: well_priorities[x]) for k, v in config["groups"].items()}

      priorities = sorted(well_priorities.values(), reverse=True)
      new_order = []
      for group in well_cycle:
          if groups[group]:
              new_order.append(groups[group].pop())
          else:
              break

      leftovers = list(set(well_priorities.keys()) - set(new_order))
      new_order += sorted(leftovers, reverse=True, key=lambda x: well_priorities[x])

      new_priorities = dict(zip(new_order, priorities))
      return new_priorities

In this piece of code, we make use of ``itertools``' ``cycle`` and ``islice`` functions.
Where ``cycle`` is used to endlessly repeat a list (in this case the "cycle" list inside ``config``)
and ``islice`` is used to limit the length of this ``cycle`` to the number of wells.

Then, well names in the various groups are sorted based on priority (highest priority last)
and the last element of a group is popped off based on the index of ``well_cycle``.

The ``leftovers`` from when the cycle can no longer be adhered to
(group has no more wells) are sorted based on priority (highest first) and added at the end of ``new_order``.

We can then store these new priorities in an ``output_filename``
and make the python file execute the code when called as follows:

.. code-block:: python

  import yaml

  def entry_point():
      config = {
        "groups": {
            "producer": ["PROD1", "PROD2", "PROD3", "PROD4", "PROD5", "PROD6", "PROD7"],
            "injector": ["INJ1", "INJ2", "INJ3"]
        }
        "cycle": ["producer", "producer", "injector"]
      }

      new_priorities = apply_cycle(well_priorities, config)

      with open(output_filename, "w") as f:
          yaml.dump(new_priorities, f, sort_keys=False)


  if __name__ == "__main__":
      entry_point()


``output_filename``:

.. code-block:: yaml

  PROD1: 10
  PROD2: 9
  INJ1: 8
  PROD3: 7
  PROD4: 6
  INJ2: 5
  PROD5: 4
  PROD6: 3
  INJ3: 2
  PROD7: 1

where you can see the "two producer, one injector" cycle has been successfully applied.
