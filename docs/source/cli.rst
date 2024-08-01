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

.. argparse::
    :module: everest.bin.everexport_script
    :func: _build_args_parser
    :prog: everexport_entry


The everest export functionality is configured in the export section of the config file.
The following represents an export section a config file set with default values.

.. code-block:: yaml

    export:
        skip_export: False
        keywords:
        batches:
        discard_gradient: True # Export only non-gradient simulations
        discard_rejected: True # Export only increased merit simulations
        csv_output_filepath: everest_output_folder/config_file.csv

When the export command `everest export config_file.yml` is run with a config file that does not define an export section default values will be used, a `config_file.csv` file in the Everest output folder will be created.
By default Everest exports only non-gradient with increased merit simulations when no config section is defined in the config file.
The file will contain optimization data for all the optimization batches and the available eclipse keywords (if a data file is available) for only the non-gradient simulations and the simulations that increase merit.

**Examples**

* Export only non-gradient simulation using the following export section in the config file

.. code-block:: yaml

    export:
        discard_rejected: False

* Export only increased merit simulation using the following export section in the config file

.. code-block:: yaml

    export:
        discard_gradient: False


* Export only a list of available batches even if they are gradient batches and if no export section is defined.

 everest export config_file.yml --batches 0 2 4

The command above is equivalent to having the following export section defined in the config file `config_file.yml`.

.. code-block:: yaml

    export:
      batches: [0, 2, 4]

* Exporting just a specific list of eclipse keywords requires the following export section defined in the config file.

.. code-block:: yaml

    export:
      keywords: ['FOIP', 'FOPT']

* Skip export by adding the following section in the config file.

.. code-block:: yaml

    export:
      skip_export: True

* Export will also be skipped if an empty list of batches is defined in the export section.

.. code-block:: yaml

    export:
      batches: []

==============
Everest `load`
==============

.. argparse::
    :module: everest.bin.everload_script
    :func: _build_args_parser
    :prog: everload_entry


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


===================
Graphical interface
===================

.. argparse::
    :module: ieverest.bin.ieverest_script
    :func: _build_args_parser
    :prog: ieverest_entry


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
