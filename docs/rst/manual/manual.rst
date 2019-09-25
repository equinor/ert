
Welcome to ERT's documentation!
===============================

ERT (Ensemble based Reservoir Tool) is a tool for model updating (history matching) using ensemble methods.
ERT is primarily developed for use with reservoir models, but the tool can be used in any area dealing
with model updates and uncertainty estimation.

Launching ERT
-------------

The most common way to launch ERT is using the graphical user interface (GUI).
Given that you have a ERT configuration file you want to run, the program is launched as follows:

.. code-block:: bash

   ert gui config.ert

To see the full documentation of the available commands and how to run the command line interface (CLI),
see :doc:`interfaces/index`.

.. toctree::
   :hidden:

   self

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting started:

   interfaces/index


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Theory:

   introduction/index


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Data types and observations:

   data_types/index
   forward_model/index
   observations/index


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Workflows:

   workflows/index
   workflows/built_in


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Configuration:

   site-configuration/index
   keywords/index


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Advanced:

   eclipse/index
   update/index
   scripting/index


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Releases:

   changes/index


.. Indices and tables
   ==================

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`
