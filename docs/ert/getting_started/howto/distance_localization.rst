Distance based localization
===========================

Distance based localization restricts parameter updates to a spatial
neighbourhood around each observation. Parameters far from a given
observation are not updated by it, which suppresses
:ref:`spurious correlations <spurious_correlations>` that arise in ensemble
methods when the ensemble size is limited.

For theoretical background, see :ref:`distance_based_localization`.

Enabling distance based localization
-------------------------------------

Distance based localization is enabled in two steps:

1. Set the update strategy for the relevant parameter types.
2. Provide location metadata for relevant observations
   see :ref:`configuring_observations_for_ert`.

Setting the update strategy
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``ANALYSIS_SET_VAR PARAMETERS`` to select the ``DISTANCE`` strategy for
spatial parameter types:

.. code-block:: none

    ANALYSIS_SET_VAR PARAMETERS FIELD DISTANCE
    ANALYSIS_SET_VAR PARAMETERS SURFACE DISTANCE

``GEN_KW`` parameters do not have spatial coordinates and cannot use
distance based localization. They can still use ``ADAPTIVE`` or the
default ``GLOBAL`` strategy, see :ref:`parameters_section`

.. code-block:: none

    ANALYSIS_SET_VAR PARAMETERS GEN_KW ADAPTIVE | GLOBAL

Providing observation locations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each observation that should participate in distance based localization
needs location metadata. See the :ref:`LOCALIZATION keyword <localization_keyword>`
reference for how to configure ``EAST``, ``NORTH``, and ``RADIUS`` on
summary, breakthrough, and RFT observations.

.. note::
    When the observation is missing location metadata, it is ignored for distance
    based localization. Nevertheless, such observations are still used for adaptive
    and global update strategies.
    Additionally, ``GENERAL_OBSERVATION`` observations do not have location metadata
    and will be excluded from distance based localization.

Full configuration example
--------------------------

.. code-block:: none

    NUM_REALIZATIONS 100

    GRID case.egrid

    FIELD PORO PARAMETER poro.grdecl INIT_FILES:poro%d.grdecl
    FIELD PERMX PARAMETER permx.grdecl INIT_FILES:permx%d.grdecl

    ANALYSIS_SET_VAR PARAMETERS FIELD DISTANCE

    OBS_CONFIG observations.txt

Where ``observations.txt`` contains observations with ``LOCALIZATION`` blocks
as described in :ref:`LOCALIZATION keyword <localization_keyword>`.

How does the distance based localization works?
-------------------------------------------------

The implemented method in ERT is based on the method published by Emerick (2016) where the Kalmain Gain is modified by multipying
elementwise by a scaling factor. The RHO matrix that contains the scaling factor has of course the same size as the Kalman Gain matrix
and can be a very large matrix for real cases with many observation (e.g. 4D seismic observations). Each element of the RHO matrix
contains the scaling factor for one pair of observation and model parameter (observation, parameter).
The Kalman Gain matrix element for the pair of (observation, parameter) is a weight factor for how much the difference
between the observed value minus the predicted value (also called innovation) is contributing to the update or change of the parameter.
By multiplying with the scaling value for that pair of observation and parameter, the effect of it is reduced.

The method is very flexible in theory. In practice, how to define the values of the scaling factor for each pair of observation
and parameter is simplified by using the lateral distance between the observation location and the parameter location
and decrease the scaling factor as function of the distance. 

In ERT the current implementation specify an influence range as a circular disk around each observations location,
and specified by a RADIUS value.

The following limitations is used in the current implementation:
- Circular influence area around each observation
- One function to calculate the scaling factor by distance is available. This is the Gaspari-Cohn correlation function.
- Only lateral distance is used.
- Same influence range from an observation regardless of which field parameter is updated.
- Same influence range from an observation regardless of which geological zone the field parameter belongs to.
- Observation specified with the GENERAL_OBSERVATION keyword is ignored when updating field parameters.
GENERAL_OBSERVATION is used when updating scalar parameters.
- Seismic 4D observations is currently not available in distance-based localization, but will come soon.

Current version will then use the observation's influence radius and for each pameter grid cell,
the algorithm computes the distance to every observation location. This distance is normalised by the observation's
radius and passed through the Gaspari-Cohn correlation function, which
produces a scaling factor between 0 and 1:

- At the observation location (distance = 0) the scaling factor is 1
  (full update).
- At a normalised distance of 2 × radius the scaling factor is 0
  (no update) according to the definition of the Gaspari-Cohn correlation function that is used.

Because only horizontal (lateral) distances are considered, vertical layers
of a 3D grid share the same scaling factor.

The localization rho matrix is computed once at the start of the analysis and reused for all iterations.
The size of the rho matrix is determined by the number of parameters and the number of observations.

Choosing a radius
-----------------

The radius of influence should reflect well's region of interest when updating parameters.
In practice:

- A smaller radius provides weaker update and reduces spurious
  correlations, but may exclude parameters that are influenced by
  the observation.
- A larger radius provides stronger ensemble update but might increase
  spurious correlations.

The default radius is set to 3000 meters.

A figure like the one below illustrates the influence range from different wells.
Each of the four wells have their own radius of influence.
The grid cell colored with blue is within the range of observation located
at  A and C, but not within the range of B and D. The grid cell colored
with red is within the range of observation in location B, C and D and
will be influenced by those in the update.

.. image:: illustrating_influence_range.png


Configuration file vs GUI
--------------------------

Distance based localization can only be enabled through the configuration
file.

For adaptive localization, the GUI provides a way to set the correlation threshold and truncation parameters.
Note that the values set in the GUI will override any values set in the configuration file.
