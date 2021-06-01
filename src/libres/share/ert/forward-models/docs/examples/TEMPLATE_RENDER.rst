Given an input file :code:`my_input.json`:

.. code-block:: json

    {
        "my_variable": "my_value"
    }

And a template file :code:`tmpl.jinja`:

.. code-block:: bash

    This is written in my file together with {{my_input.my_variable}}

This job will produce an output file:

.. code-block:: bash

    This is written in my file together with my_value

By invoking the :code:`FORWARD_MODEL` as such:

.. code-block:: bash

    FORWARD_MODEL TEMPLATE_RENDER(<INPUT_FILES>=my_input.json, <TEMPLATE_FILE>=tmpl.jinja, <OUTPUT_FILE>=output_file)
