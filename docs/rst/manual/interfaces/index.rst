Running ERT
===========

Graphical User Interface (GUI)
------------------------------

.. argparse::
   :module: ert_gui.main
   :func: get_ert_parser
   :prog: ert
   :path: gui

Command Line Interface (CLI)
----------------------------

The following sub commands can be used in order to use Ert's command line interface.
Note that different sub commands may require different additional arguments.

Test Run
~~~~~~~~

.. argparse::
   :module: ert_gui.main
   :func: get_ert_parser
   :prog: ert
   :path: test_run

Ensemble Experiment
~~~~~~~~~~~~~~~~~~~

.. argparse::
   :module: ert_gui.main
   :func: get_ert_parser
   :prog: ert
   :path: ensemble_experiment

Ensemble Smoother
~~~~~~~~~~~~~~~~~

.. argparse::
   :module: ert_gui.main
   :func: get_ert_parser
   :prog: ert
   :path: ensemble_smoother

ES MDA
~~~~~~

.. argparse::
   :module: ert_gui.main
   :func: get_ert_parser
   :prog: ert
   :path: es_mda
