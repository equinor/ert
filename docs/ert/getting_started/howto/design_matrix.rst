Design matrix
=============

The `DESIGN_MATRIX` keyword in ERT allows users to define a design matrix in Excel,
as described in the :ref:`design_matrix` section of the documentation. The design matrix
contains parameters and their values per realization.

`DESIGN2PARAMS` is a deprecated forward model and users are encouraged to migrate to the
`DESIGN_MATRIX` keyword for defining design matrices. The `DESIGN_MATRIX` keyword accepts the same
parameters as `DESIGN2PARAMS`.

If the same parameter name is defined both in the design matrix and in the GEN_KW parameter definition,
the value from the design matrix takes precedence and the parameter will not be updated by ERT.

Migration from `DESIGN2PARAMS`
------------------------------

Migrating from the `DESIGN2PARAMS` forward model involves updating your configuration to use the `DESIGN_MATRIX` keyword.
The `DESIGN_MATRIX` keyword replaces `DESIGN2PARAMS` by integrating design matrix functionality directly into ERT,
offering improved validation and parameter usability (like plotting) in ERT.

To migrate, replace instances of `DESIGN2PARAMS` in your configuration file with `DESIGN_MATRIX`:

For example

::

    FORWARD_MODEL DESIGN2PARAMS(<xls_filename>=<CONFIG_PATH>/<DESIGN_MATRIX>, <designsheet>=<DESIGN_SHEET>, <defaultssheet>=DefaultValues)

should be replaced with DESIGN_MATRIX as follows:

::

    DESIGN_MATRIX <CONFIG_PATH>/<DESIGN_MATRIX> DESIGN_SHEET:<DESIGN_SHEET> DEFAULT_SHEET:DefaultValues

Additionally, review the documentation for :ref:`DESIGN_MATRIX <design_matrix>`.

NOTE: The DESIGN_MATRIX validation is more strict than that of DESIGN2PARAMS.
Some problems that were previously hidden or resulting in failures during runtime, will now result
in errors during validation of the configuration:

- Missing values (empty cells) in the design matrix
- Duplicate parameter names in the design matrix
- Design sheet is empty
- REAL column must only contain unique, positive integers

Migration from `DESIGN_KW`
--------------------------

Migration to RUN_TEMPLATE is straightforward.
The `RUN_TEMPLATE` keyword replaces the `DESIGN_KW` forward model by doing magic string replacements in the file directly, without
going through a `parameters.txt` file. `RUN_TEMPLATE` also copies the file automatically to run path.

To migrate, replace instances of `DESIGN_KW` in your configuration file with `RUN_TEMPLATE`:

For example

::

    FORWARD_MODEL DESIGN_KW(<template_file>=my_text_file_template.txt, <result_file>=my_text_file.txt)

should be replaced with RUN_TEMPLATE as follows:

::

    RUN_TEMPLATE my_text_file_template.txt my_text_file.txt

Additionally, review the documentation for :ref:`RUN_TEMPLATE <run_template>`.
