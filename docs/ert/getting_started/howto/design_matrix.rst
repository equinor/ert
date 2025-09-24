Design matrix
=============

Migration from DESIGN2PARAMS
----------------------------

Migrating from the `DESIGN2PARAMS` forward model involves updating your configuration to use more ert aware `DESIGN_MATRIX` keyword.
The `DESIGN_MATRIX` keyword replaces `DESIGN2PARAMS` by allowing users to define the relationship between design model parameters in a matrix format, offering improved validation and clarity.
To migrate, replace instances of `DESIGN2PARAMS` in your configuration file with `DESIGN_MATRIX`:

For example

::

    FORWARD_MODEL DESIGN2PARAMS(<xls_filename>=<CONFIG_PATH>/<DESIGN_MATRIX>, <designsheet>=<DESIGN_SHEET>, <defaultssheet>=DefaultValues)

will be replaced in the following way:

::

    DESIGN_MATRIX <CONFIG_PATH>/<DESIGN_MATRIX> DESIGN_SHEET:<DESIGN_SHEET> DEFAULT_SHEET:DefaultValues

Additionally, review the documentation for :ref:`DESIGN_MATRIX <design_matrix>` to understand its syntax and capabilities, as it provides enhanced validation compared to the `DESIGN2PARAMS` model.
