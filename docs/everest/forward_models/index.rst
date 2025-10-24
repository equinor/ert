
.. _cha_forward_model_jobs:


******************
Forward model jobs
******************

.. toctree::
    :hidden:

    builtin
    custom

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
