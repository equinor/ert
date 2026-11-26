Running ERT
===========

Graphical user interface (GUI)
------------------------------

.. argparse::
   :module: ert.__main__
   :func: get_ert_parser
   :prog: ert
   :path: gui

.. _one_gui_per_storage:

One GUI per storage folder
~~~~~~~~~~~~~~~~~~~~~~~~~~

Only one ERT GUI can use a storage folder, set by :ref:`ENSPATH <enspath>`, at
a time. This also applies to different configuration files that share an
ENSPATH. For example, ``poly.ert`` and ``poly_new.ert`` in the same folder share
the default storage folder ``storage`` when neither sets ENSPATH. While
``ert gui poly.ert`` is open, running

.. code-block:: shell

   ert gui poly_new.ert

prints an error and exits without opening a window:

.. code-block:: text

   Error: Server already running for project at /home/user/poly_example/storage
   Cannot connect to storage service because it is already running.

Close the open ERT GUI before you start ERT again with a configuration that
uses the same storage folder. If ERT has just crashed, wait a few seconds and
try again. To have both GUIs open at the same time, set a different ENSPATH in
one of the configuration files.

Command line interface (CLI)
----------------------------

The following sub commands can be used in order to use ERT's command line interface.
Note that different sub commands may require different additional arguments.

Test run
~~~~~~~~

.. argparse::
   :module: ert.__main__
   :func: get_ert_parser
   :prog: ert
   :path: test_run

Ensemble experiment
~~~~~~~~~~~~~~~~~~~

.. argparse::
   :module: ert.__main__
   :func: get_ert_parser
   :prog: ert
   :path: ensemble_experiment

ES MDA
~~~~~~

.. argparse::
   :module: ert.__main__
   :func: get_ert_parser
   :prog: ert
   :path: es_mda

Workflow
~~~~~~~~

.. argparse::
   :module: ert.__main__
   :func: get_ert_parser
   :prog: ert
   :path: workflow

Login
~~~~~

Log in with the login provider installed by a site plugin, see
:doc:`../getting_started/howto/plugin_system`. Fails if no login provider is installed.

.. argparse::
   :module: ert.__main__
   :func: get_ert_parser
   :prog: ert
   :path: login
