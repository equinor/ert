.. _cha_forward_model_jobs:




******************
Forward model jobs
******************

Some of the forward models are connected and often the output of one is the
input to another. However, they maintain the same format, and thus a subset
of the jobs can be run independently. For example, if well_constraints is not
necessary, it can be omitted, and if well_order is not a concern, the input to
the well_constraints can be setup and copied (in the everest configuration,
rather than using the drill_planner every time).
Some of them are however mandatory, e.g. the add_templates is a prerequisite
for the schmerge job.

**Example**

.. code-block:: yaml

    forward_model:
      - fm_drill_planner -i well.json
                         -c drill_planner_config.yaml
                         -opt optimizer_values.yml
                         -o wells_dp_result.json
      - fm_well_constraints -i wells_dp_result.json
                            -c well_constraint_config.yml
                            -rc rate_input.json
                            -pc phase_input.json
                            -dc duration_input.json
                            -o wells_wc_result.json
      - fm_add_templates -c template_config.yml
                         -i wells_wc_result.json
                         -o wells_tmpl_result.json
      - fm_schmerge  -s raw_schedule.sch
                     -i wells_tmpl_result.json
                     -o result_schedule.sch



.. everest_forward_model::


Template rendering
==================

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


.. _eclipse100:

Eclipse simulator
=================

.. code-block:: bash

  eclipse100 <eclbase> --version <version_number>

Running eclipse with parallel option
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is possible to run eclipse with multiple CPUs on clusters. This requires the eclipse data file to have the
parallel option and the everest config needs to specify the number of CPUs per node:

.. code-block:: yaml

  simulator:
    cores_per_realization: x

where x is an int giving the number of cores. The eclipse100 forward model also needs to be given the argument to use
multiple cores:

.. code-block:: bash

  eclipse100 <eclbase> --version <version_number> --num-cpu x

where x is the number of cores.

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
^^^^^^^^^^^^^^^^^^^^^^^
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
