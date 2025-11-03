
Built-in forward model jobs
===========================


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

Everest supports built-in forward model steps for running common reservoir simulators
in optimization workflows related to subsurface activities.
This includes **OPM Flow**, **Eclipse100**, and **Eclipse300**.

.. note::

    Everest does not include the simulators themselves. Any use of simulators **OPM Flow**,
    **Eclipse100**, or **Eclipse300**, requires respective installation of that simulator
    available in your environment. Everest interfaces with these simulators but does not
    bundle or install them. Ensure they are accessible via your system's ``$PATH``
    or configured appropriately in your environment.

Supported simulator jobs
~~~~~~~~~~~~~~~~~~~~~~~~

You can specify the following simulator jobs in your Everest configuration:

- ``flow`` — runs OPM Flow.
- ``eclipse100`` — runs Eclipse 100 (i.e., ``eclipse``).
- ``eclipse300`` — runs Eclipse 300 (i.e., ``e300``).

All three map internally to the same reservoir simulator runner, but differ in how
arguments are interpreted and which simulator is launched.

.. _flow:

Flow usage
~~~~~~~~~~

Everest will run the Flow simulator using either the ``flow`` binary or a wrapper script
called ``flowrun``, depending on what is available in the user's environment (``$PATH``).

- If ``flowrun`` is found, it takes precedence and enables additional features such as:

  - version selection (``--version``)
  - parallel execution (``--np``, ``--threads``)
  - default flags (e.g., ``--enable-esmry=true``)

- If only ``flow`` is available, Everest will invoke it directly with the provided arguments.

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

If your environment does **not** include a ``flowrun`` wrapper, Everest will invoke the ``flow`` binary directly.
In this case, Everest does **not** insert ``mpirun`` or manage parallel execution.
You must handle MPI launching manually by including ``mpirun`` in the job line and install ``mpirun``
as a custom forward model job in Everest via the ``install_jobs`` section.

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

- Installs ``mpirun`` as a custom job in Everest
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

Everest usage example
~~~~~~~~~~~~~~~~~~~~~
The following illustrates an example of a forward model section of an Everest config file:

.. code-block:: yaml

    forward_model:
      - well_constraints  -i files/well_readydate.json -c files/wc_config.yml -rc well_rate.json -o wc_wells.json
      - add_templates     -i wc_wells.json -c files/at_config.yml -o at_wells.json
      - schmerge          -s eclipse/include/schedule/schedule.tmpl -i at_wells.json -o eclipse/include/schedule/schedule.sch
      - job: eclipse100   r{{ eclbase }} --version 2020.2
        results:
          file_name: r{{eclbase}}
          type: summary
          keys: ['FOPR', 'WOPR']
      - rf                -s r{{ eclbase }} -o rf

The ``add_templates`` job does **NOT** need to be *installed* it is already part of the default everest jobs.
In the example above all files present in the ``files`` folder need to be provided by the user. The ``files``
folder should have the following structure:

.. code-block:: yaml

 files/
    |- well_readydate.json
    |- wc_config.yml
    |- at_config.yml
    |- templates/
        |- wconinje.j2.html
        |- wconprod.j2.html

and should be *installed* in the everest config file:

.. code-block::

    install_data:
      -
        source: r{{ configpath }}/../input/files
        target: files
        link: true

``well_readydate.json``

.. code-block:: json

    [
       {
         "name": "PROD1",
         "readydate": "2000-01-01",
       },
       {
         "name": "PROD2",
         "readydate": "2000-01-01",
       },
       {
         "name": "INJECT1",
         "readydate": "2000-01-01",
       },
       {
         "name": "INJECT2",
         "readydate": "2000-01-01",
       }
    ]

``wc_config.yml``

.. code-block:: yaml

    PROD1:
      1:
        phase:
          value: OIL
        duration:
          value: 50
    PROD2:
      1:
        phase:
          value: OIL
        duration:
          value: 50
    INJECT1:
      1:
        phase:
          value: WATER
        duration:
          value: 50
    INJECT2:
      1:
        phase:
          value: WATER
        duration:
          value: 50

``at_config.yml``

.. code-block:: yaml

    templates:
      -
        file: './files/templates/wconinje.j2.html'
        keys:
            opname: rate
            phase: WATER
      -
        file: './files/templates/wconprod.j2.html'
        keys:
            opname: rate
            phase: OIL

``wconprod.j2.html``

.. code-block:: jinja

    WCONPROD
      '{{ name }}'  'OPEN'  'ORAT' {{ rate }}   4* 100   /
    /

``wconinje.j2.html``

.. code-block:: jinja

    WCONINJE
      '{{ name }}'  '{{ phase }}'  'OPEN'  'RATE' {{ rate }}   1* 320  1*  1*    1*   /
    /

In the above example of the forward model section of the config file:

* The file ``wc_wells.json`` is a direct output of the ``well_constraint`` job.
* The ``add_templates`` job uses the same file ``wc_wells.json`` as an input for the job.
* The ``wc_wells.json`` file is not modified by the user. Any modification to this file should be done using a custom job (see the section :ref:`cha_creating_custom_jobs` for more information on how to do that).

If the file is to be modified by a custom job, the everest config should contain:

.. code-block:: yaml

    install_jobs:
      -
        name: custom_job
        executable: jobs/custom_job.exe

    forward_model:
      - well_constraints  -i files/well_readydate.json -c files/wc_config.yml -rc well_rate.json -o wc_wells.json
      - custom_job        -i wc_wells.json -o wc_wells_custom.json
      - add_templates     -i wc_wells_custom.json -c files/at_config.yml -o at_wells.json
      - schmerge          -s eclipse/include/schedule/schedule.tmpl -i at_wells.json -o eclipse/include/schedule/schedule.sch
      - job: eclipse100   r{{ eclbase }} --version 2020.2
        results:
          file_name: r{{eclbase}}
          type: summary
          keys: ['FOPR', 'WOPR']
      - rf                -s r{{ eclbase }} -o rf


``wc_wells.json``

.. code-block:: json

    [
      {
        "name": "PROD1",
        "readydate": "2000-01-01",
        "ops": [
          {
            "phase": "OIL",
            "rate": 550.0015,
            "date": "2000-01-01",
            "opname": "rate"
          }
        ]
      },
      {
        "name": "PROD2",
        "readydate": "2000-01-01",
        "ops": [
          {
            "phase": "OIL",
            "rate": 860.0048,
            "date": "2000-01-01",
            "opname": "rate"
          }
        ]
      },
      {
        "name": "INJECT1",
        "readydate": "2000-01-01",
        "ops": [
          {
            "phase": "WATER",
            "rate": 5499.93,
            "date": "2000-01-01",
            "opname": "rate"
          }
        ]
      },
      {
        "name": "INJECT2",
        "readydate": "2000-01-01",
        "ops": [
          {
            "phase": "WATER",
            "rate": 5500.075,
            "date": "2000-01-01",
            "opname": "rate"
          }
        ]
      }
    ]

The add_templates job will search in the file ``wc_wells.json`` for the keys defined by the user in the config file ``at_config.yml``
and where the keys are present the job will add the corresponding template file.  The resulting output ``at_wells.json`` has the following form:

``at_wells.json``

.. code-block:: json

    [
      {
        "name": "PROD1",
        "readydate": "2000-01-01",
        "ops": [
          {
            "phase": "OIL",
            "rate": 550.0015,
            "date": "2000-01-01",
            "opname": "rate",
            "template": "./files/templates/wconprod.j2.html"
          }
        ]
      },
      {
        "name": "PROD2",
        "readydate": "2000-01-01",
        "ops": [
          {
            "phase": "OIL",
            "rate": 860.0048,
            "date": "2000-01-01",
            "opname": "rate",
            "template": "./files/templates/wconprod.j2.html"
          }
        ]
      },
      {
        "name": "INJECT1",
        "readydate": "2000-01-01",
        "ops": [
          {
            "phase": "WATER",
            "rate": 5499.93,
            "date": "2000-01-01",
            "opname": "rate",
            "template": "./files/templates/wconinje.j2.html"
          }
        ]
      },
      {
        "name": "INJECT2",
        "readydate": "2000-01-01",
        "ops": [
          {
            "phase": "WATER",
            "rate": 5500.075,
            "date": "2000-01-01",
            "opname": "rate",
            "template": "./files/templates/wconinje.j2.html"
          }
        ]
      }
    ]

Next, the ``at_wells.json`` file is used as an input for the schedule merge job ``schmerge`` together with the initial schedule template
``schedule.tmpl`` file, which will result in the new schedule file ``schedule.sch`` used for the simulation.

For the following entry in the ``at_wells.json``:

.. code-block:: json

      {
        "name": "PROD1",
        "readydate": "2000-01-01",
        "ops": [
          {
            "phase": "OIL",
            "rate": 550.0015,
            "date": "2000-01-01",
            "opname": "rate",
            "template": "./files/templates/wconprod.j2.html"
          }
        ]
      }

and the template ``wconprod.j2.html``:

.. code-block:: jinja

    WCONPROD
      '{{ name }}'  'OPEN'  'ORAT' {{ rate }}   4* 100   /
    /

the resulting entry in ``schedule.sch`` is as follows:

.. code-block::

    DATES
     01 JAN 2000 / --ADDED
    /

    --start ./files/templates/wconprod.j2.html
    WCONPROD
      'PROD1'  'OPEN'  'ORAT' 550.0015   4* 100   /
    /

    --end ./files/templates/wconprod.j2.html

where ``"--"`` marks the beginning of a comment line and will be ignored by the simulator.


Other template examples
-----------------------
The `jinja2 <https://jinja.palletsprojects.com/>`_ templating language is supported by
the schedule merge job, and can be used to write the templates.
Below a few default examples can be found:

**Water injection template**

.. code-block:: jinja

    WCONINJE
      '{{ name }}' '{{ phase }}' 'OPEN' 'RATE' {{ rate }} 5*   /
    /

**Gas production template**

.. code-block:: jinja

    WCONPROD
      '{{ name }}' 'OPEN' 'GRAT' {{ rate }}  5*   /
    /

**Oil production template**

.. code-block:: jinja

    WCONPROD
      '{{ name }}' 'OPEN' 'ORAT' {{ rate }}  5*  /
    /

**Well open template**

.. code-block:: jinja

    WELOPEN
      '{{ name }}' 'OPEN' /
    /

More information regarding template design and usage can be found `here <https://jinja.palletsprojects.com/templates/>`_.
