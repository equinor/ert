
.. _cha_builtin_forward_model_jobs:

***************************
Built-in forward model jobs
***************************


Template rendering
------------------

.. argparse::
   :module: ert.resources.forward_models.template_render
   :func: _build_argument_parser
   :prog: template_render

Example
~~~~~~~

Given an input file ``my_input.json``:

.. code-block:: json

   {
       "my_variable": "my_value"
   }

And a template file ``tmpl.jinja``:

.. code-block:: jinja

    This is written in my file together with {{my_input.my_variable}}

Run the script with:

.. code-block:: bash

    template_render -i my_input.json -t tmpl.jinja -o my_output.txt

This will produce an output file with the content:

.. code-block:: text

    This is written in my file together with my_value


.. _built_in_reservoir_simulators:

Reservoir simulators (Flow & Eclipse)
-------------------------------------

EVEREST supports built-in forward model steps for running common reservoir simulators
in optimization workflows related to subsurface activities.
This includes **OPM Flow**, **Eclipse100**, and **Eclipse300**.

.. note::

    EVEREST does not include the simulators themselves. Any use of simulators **OPM Flow**,
    **Eclipse100**, or **Eclipse300**, requires respective installation of that simulator
    available in your environment. EVEREST interfaces with these simulators but does not
    bundle or install them. Ensure they are accessible via your system's ``$PATH``
    or configured appropriately in your environment.

Supported simulator jobs
~~~~~~~~~~~~~~~~~~~~~~~~

You can specify the following simulator jobs in your EVEREST configuration:

- ``flow`` — runs OPM Flow.
- ``eclipse100`` — runs Eclipse 100 (i.e., ``eclipse``).
- ``eclipse300`` — runs Eclipse 300 (i.e., ``e300``).

All three map internally to the same reservoir simulator runner, but differ in how
arguments are interpreted and which simulator is launched.

.. _flow:

Flow usage
~~~~~~~~~~

EVEREST will run the Flow simulator using either the ``flow`` binary or a wrapper script
called ``flowrun``, depending on what is available in the user's environment (``$PATH``).

- If ``flowrun`` is found, it takes precedence and enables additional features such as:

  - version selection (``--version``)
  - parallel execution (``--np``, ``--threads``)
  - default flags (e.g., ``--enable-esmry=true``)

- If only ``flow`` is available, EVEREST will invoke it directly with the provided arguments.

You can check which binary is used by running ``which flowrun`` or ``which flow`` in your terminal.

Single-threaded Flow example
""""""""""""""""""""""""""""

.. code-block:: yaml

   forward_model:
     - job: flow r{{ eclbase }}
       results:
         file_name: r{{ eclbase }}
         type: summary

Multi-process and multi-threaded Flow example
"""""""""""""""""""""""""""""""""""""""""""""

.. code-block:: yaml

   forward_model:
     - job: flow r{{ eclbase }} --np 8 --threads 4 --version stable
       results:
         file_name: r{{ eclbase }}
         type: summary
         keys: ["FOPR", "WOPR"]

This runs Flow with 8 MPI ranks, each using 4 OpenMP threads. The version ``stable`` is selected
(if supported by the wrapper). Additional Flow arguments can be passed as needed.

Manual MPI launch (without flowrun wrapper)
"""""""""""""""""""""""""""""""""""""""""""

If your environment does **not** include a ``flowrun`` wrapper, EVEREST will invoke the ``flow`` binary directly.
In this case, EVEREST does **not** insert ``mpirun`` or manage parallel execution.
You must handle MPI launching manually by including ``mpirun`` in the job line and install ``mpirun``
as a custom forward model job in EVEREST via the ``install_jobs`` section.

.. code-block:: yaml

   install_jobs:
     -
       name: mpirun
       executable: /usr/bin/mpirun

   forward_model:
     - job: mpirun -np 8 flow r{{ eclbase }}.DATA --threads-per-process=4
       results:
         file_name: r{{ eclbase }}
         type: summary
         keys: ["FOPR", "WOPR"]

This example:

- Installs ``mpirun`` as a custom job in EVEREST
  - NOTE: executable (path) should point to the ``mpirun`` binary in your environment (check ``which mpirun``)
- Launches Flow with ``mpirun -np 8`` (8 MPI ranks)
- Sets 4 OpenMP threads per rank using Flow's native flag ``--threads-per-process=4``
- Assumes ``mpirun`` and ``flow`` are available in the environment

.. _eclipse100:
.. _eclipse300:

Eclipse100 and Eclipse300 usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To run Eclipse100, use the following syntax:

.. code-block:: yaml

   forward_model:
     - job: eclipse100 r{{ eclbase }} --version 2020.2
       results:
         file_name: r{{ eclbase }}
         type: summary
         keys: ["FOPR", "WOPR"]

Required and optional arguments:

- ``--version <VERSION>``: **Required** for Eclipse jobs. Specifies the simulator version.
- ``-i / --ignore-errors``: Continue even if the simulator returns an error.
- ``--summary-conversion``: Enables summary conversion (only available for Eclipse).

To run Eclipse300, please use the following syntax:

.. code-block:: yaml

   forward_model:
     - job: eclipse300 r{{ eclbase }} --version 2021.1 --summary-conversion
       results:
         file_name: r{{ eclbase }}
         type: summary
         keys: ["FOPT", "FWPT"]

These arguments are passed to the simulator runner and used to construct the command:

.. code-block:: text

   eclrun -v 2021.1 e300 <deckfile> --summary-conversion yes

The deck file is automatically resolved from the base name (e.g., ``r{{ eclbase }}.DATA``).

Running Eclipse in parallel
"""""""""""""""""""""""""""

To run Eclipse simulators (``eclipse``, ``e300``) in parallel, you must include the ``PARALLEL`` keyword in the ``RUNSPEC`` section of your simulation deck.
The number of MPI processes is determined internally by Eclipse based on the deck configuration, not by command-line options (i.e., ``--np``).
If ``PARALLEL`` is missing, the simulation runs in serial mode regardless of ``--np``.

While Eclipse determines parallelism internally, the job scheduler (e.g., SLURM, LSF) may allocate resources based on
``cores_per_node``, (see the example below). This affects how many MPI ranks are launched if the runner
or wrapper respects the allocation. However, Eclipse itself still relies on the deck configuration
to determine actual parallel behavior.

.. code-block:: yaml

    simulator:
      cores_per_node: 16

    forward_model:
      - job: eclipse300 r{{ eclbase }} --version 2021.1
        results:
          file_name: r{{ eclbase }}
          type: summary
