.. _cha_cli:

**********************
Command Line Interface
**********************

===============
``everest``
===============

.. argparse::
    :module: everest.bin.main
    :func: _build_args_parser
    :prog: start_everest


===============
``everest run``
===============

.. argparse::
    :module: everest.bin.everest_script
    :func: _build_args_parser
    :prog: everest_entry


===================
``everest monitor``
===================

.. argparse::
    :module: everest.bin.monitor_script
    :func: _build_args_parser
    :prog: monitor_entry


================
``everest kill``
================

.. argparse::
    :module: everest.bin.kill_script
    :func: _build_args_parser
    :prog: kill_entry


================
``everest lint``
================

.. argparse::
    :module: everest.bin.everlint_script
    :func: _build_args_parser
    :prog: lint_entry


==================
``everest render``
==================

.. argparse::
    :module: everest.bin.everconfigdump_script
    :func: _build_args_parser
    :prog: config_dump_entry


.. _ev_branch:

==================
``everest branch``
==================

.. argparse::
    :module: everest.bin.config_branch_script
    :func: _build_args_parser
    :prog: config_branch_entry


===================
``everest results``
===================

.. argparse::
    :module: everest.bin.visualization_script
    :func: _build_args_parser
    :prog: visualization_entry


==================
``everest export``
==================

The ``everest export`` command has been removed. All data is now always exported to the optimization output directory.

================
``everest gui``
================

The ``everest gui`` has been removed. Please use ``everest run --gui`` instead.
