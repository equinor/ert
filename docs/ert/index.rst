.. image:: ../../src/ert/gui/resources/gui/img/ert_icon.svg
   :align: center
   :scale: 50%

Welcome to the ERT documentation!
=================================

ERT (Ensemble based Reservoir Tool) is a free and open-source tool for automating complex workflows,
such as uncertainty quantification and data assimilation.
It is heavily used in the petroleum industry for reservoir management and production optimization,
but ERT is a general tool and is used in other domains, such as wind-farm management and Carbon Capture and Storage (CCS).

If you're new to ERT:

1. Begin by ensuring you've correctly installed it.
   Check out the :doc:`getting_started/setup` guide for assistance.
2. Follow the :doc:`getting_started/configuration/poly_new/guide` to learn how to use ERT for parameter estimation.

To understand the theoretical foundations of ensemble-based methods, head over to :doc:`theory/ensemble_based_methods`.

.. toctree::
   :hidden:

   self

.. toctree::
   :hidden:
   :caption: Getting started

   getting_started/setup
   getting_started/configuration/poly_new/guide
   getting_started/howto/esmda_restart
   getting_started/howto/adaptive_localization.ipynb
   getting_started/updating_parameters/index
   getting_started/howto/plugin_system
   getting_started/webviz-ert/webviz-ert

.. toctree::
   :hidden:
   :caption: Reference

   reference/running_ert
   reference/configuration/index
   reference/workflows/index

.. toctree::
   :hidden:
   :caption: Theory

   theory/ensemble_based_methods

.. toctree::
   :hidden:
   :caption: Developer Documentation

   developer_documentation/roadmap
   developer_documentation/dev-strategy
   developer_documentation/concepts
   developer_documentation/forward_model
   developer_documentation/qt


.. toctree::
   :hidden:
   :caption: About

   about/index
   PyPI releases <https://pypi.org/project/ert/>
   Code in GitHub <https://github.com/equinor/ert>
   Issue tracker <https://github.com/equinor/ert/issues>

.. Indices and tables
   ==================

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`
