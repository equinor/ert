Running ERT
===========

Graphical user interface (GUI)
------------------------------

.. argparse::
   :module: ert.__main__
   :func: get_ert_parser
   :prog: ert
   :path: gui

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

Ensemble smoother
~~~~~~~~~~~~~~~~~

.. argparse::
   :module: ert.__main__
   :func: get_ert_parser
   :prog: ert
   :path: ensemble_smoother

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
