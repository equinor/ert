.. _cha_cli:

**********************
Command Line Interface
**********************

After setting up a ``*.yml`` config file describing the optimization case required the reservoir planning optimization tool Everest supports the following commands.

**Description**

The main entry point of the Everest application

**Command**

.. code-block:: bash

  everest [<command>] [optional_arguments]

When no `<command>` is given, Everest supports the following optional arguments

--docs     Display in the console the Everest config file documentation
--manual   Display in the console the Everest config file extended documentation
--version  Show installed Everest version


==============
 Everest `run`
==============
**Description**

Start an optimization case based on a given config file

**Command**

 everest run  [optional_arguments] <config_file>

**Examples**

* Start an optimization case

 everest run config_file.yml

Closing the console used to start the optimization case while it is running will not terminate the optimization process.

Using the command `everest monitor config_file.yml`,  while the optimization case is still running and without modifying, the config file will reattach to the optimization case and display the optimization progress from the point it has reached. To stop a running optimization use `everest kill config_file.yml`.

* Stop a running optimization case

 everest kill config_file.yml

* Run again a finished(or stopped) optimization case

 everest run --new-run config_file.yml

==================
 Everest `monitor`
==================
**Description**

Attach to a running optimization case based on a given config file

**Command**

 everest monitor  [optional_arguments] <config_file>

**Examples**

* Attach to a running case and monitor the progress

 everest monitor config_file.yml

Closing the console while monitoring the optimization case will not terminate the optimization process.

Using again the command `everest monitor config_file.yml`, will reattach to the optimization case and display the optimization progress from the point it has reached.

==================
 Everest `kill`
==================
**Description**

Kill a running optimization case based on a given config file

**Command**

 everest kill  <config_file>

**Examples**

* Kill a running case

 everest kill config_file.yml

.. _evexport:

================
Everest `export`
================

Export data from a completed optimization case
**Command**

 everest export <config_file> [optimal_arguments]

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

The file will contain optimization data for all the optimization batches and the available eclipse keywords (if a data file is available) for only the non-gradient simulations and the simulations that increase merit.

**Optional arguments**

The Everest export functionality support one additional optional argument

 -b, --batches # The list of batches that will be export to the csv file.

**Examples**

* By default Everest exports only non-gradient with increased merit simulations when no config section is defined in the config file.

 everest export config_file.yml

* Export only non-gradient simulation using the following export section in the config file

.. code-block:: yaml

    export:
        discard_rejected: False

* Export only increased merit simulation using the following export section in the config file

.. code-block:: yaml

    export:
        discard_gradient: False

* Export all available simulations using the following export section in the config file

.. code-block:: yaml

    export:
        discard_gradient: False
        discard_rejected: False

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

**Description**

Load Eclipse data from an existing simulation folder

**Command**
 everest load  <config_file> [optional_arguments]

**Optional arguments**

.. code-block:: yaml

    -s, --silent
    --overwrite
    -b, --batches

**Examples**

* Load Eclipse data for existing simulation folder.
  By default, the user will be requested to confirm the action.

 everest load config_file.yml

* Silently load the Eclipse data while also backing up the existing simulation folder

 everest load config_file.yml -s

or

 everest load config_file.yml  --silent

* Silently load the Eclipse data without backing up the existing simulation folder

 everest load config_file.yml  -s --overwrite

or

 everest load config_file.yml --silent --overwrite


* Load eclipse data only for specific simulation baches

 everest load config_file.yml --batches 0 1 3 5

or

 everest load config_file.yml -b 0 1 3 5

==============
Everest `lint`
==============

**Description**

Validate a config file

**Command**

 everest lint <config_file>


**Example**

* Check if `config_file.yml` is a valid Everest config file and no errors are present

 everest lint config_file.yml

================
Everest `render`
================

**Description**

Display the configuration data loaded from a config file after replacing templated arguments.

**Command**

 everest render <config_file>

**Example**

* Show loaded configuration data

 everest render config_file.yml

===================
Graphical interface
===================

Start the Everest graphical user interface

**Command**

 everest gui [config_file]

**Example**

* Start everest GUI without loading any prior config file

 everest gui

* Start everest GUI by loading existing config file

 everest gui config_file.yml

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

Start the everest visualization plugin. If no visualization plugin is installed the message:
``No visualization plugin installed!`` will be displayed in the console.

**Command**

.. code-block:: yaml

 usage: everest results <config_file.yml>

Because visualization plugins require optimization data, the command above should only be called for in-progress or finished
optimization cases.

Positional Arguments
====================
``config_file``

The path to the everest configuration file


Plugin
======

The recommended open-source everest visualization plugin is Everviz_

.. _Everviz: https://github.com/equinor/everviz


It can be installed using `pip`

.. code-block:: yaml

 pip install everviz
