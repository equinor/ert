.. _cha_cli:

**********************
Command Line Interface
**********************

.. argparse::
    :module: everest.bin.main
    :func: _build_args_parser
    :prog: start_everest

==============
 Everest `run`
==============

.. argparse::
    :module: everest.bin.everest_script
    :func: _build_args_parser
    :prog: everest_entry

Closing the console used to start the optimization case while it is running will not terminate the optimization process.
To continue monitoring the running optimization use the command `everest monitor config_file.yml`.
To stop a running optimization use `everest kill config_file.yml`.


==================
 Everest `monitor`
==================

.. argparse::
    :module: everest.bin.monitor_script
    :func: _build_args_parser
    :prog: monitor_entry

Closing the console while monitoring the optimization case will not terminate the optimization process.

Using again the command `everest monitor config_file.yml`, will reattach to the optimization case and display the optimization progress from the point it has reached.

==================
 Everest `kill`
==================

.. argparse::
    :module: everest.bin.kill_script
    :func: _build_args_parser
    :prog: kill_entry


.. _evexport:

================
Everest `export`
================

The everest export has been removed. All data is now always exported to the optimization output directory.

==============
Everest `lint`
==============

.. argparse::
    :module: everest.bin.everlint_script
    :func: _build_args_parser
    :prog: lint_entry

================
Everest `render`
================

.. argparse::
    :module: everest.bin.everconfigdump_script
    :func: _build_args_parser
    :prog: config_dump_entry


.. _ev_branch:

==============
Everest branch
==============

.. argparse::
    :module: everest.bin.config_branch_script
    :func: _build_args_parser
    :prog: config_branch_entry

**Description**

The *everest branch* command is designed to help users quickly create a new config file based on a previous config file
used in an optimization experiment.

The user is required to provide an existing batch number form the previous optimization experiment.

The newly created config file will contain updated values for the control's initial guesses.

The new values for the control's initial guess will be the control values associated with the given batch number
in the previous optimization case.

**Warning**
Removing the optimization output folder before running the *branch* will make the command unable to successfully execute

The *branch* command does not provide optimization experiment restart functionality. Starting an optimization case based
on the newly created config file is considered an new optimization experiment.

===============
Everest results
===============

.. argparse::
    :module: everest.bin.visualization_script
    :func: _build_args_parser
    :prog: visualization_entry


If no visualization plugin is installed the message:
``No visualization plugin installed!`` will be displayed in the console.


Plugin
======

The recommended open-source everest visualization plugin is Everviz_

.. _Everviz: https://github.com/equinor/everviz


It can be installed using `pip`

.. code-block:: yaml

 pip install everviz
