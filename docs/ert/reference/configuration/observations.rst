.. _Configuring_observations_for_ERT:

Observations
============


General overview
----------------

When using ERT to condition on dynamic data, it is necessary to
specify the data/observations to be used. For every piece of data
ERT needs to know:

 - The measured value of the data.
 - The uncertainty (standard deviation) of the measured data.
 - The time of measurement.
 - How to simulate a response of the data given a parameterized forward model.

This information is configured in an observation file. The name/path
to this observation file is declared in the main ERT config file using the
:ref:`OBS_CONFIG <obs_config>` keyword.

The observation file is a plain text file, and is in essence built around three
different classes of observations using the associated keywords:

 - :ref:`SUMMARY_OBSERVATION <summary_observation>`: For explicitly giving
   scalar observation values for responses that can be extracted from a
   reservoir simulator :term:`summary files`. Examples are rates from separator
   tests, water cut, GOR, shut in pressures, etc.

 - :ref:`GENERAL_OBSERVATION <general_observation>`: All other observations.
   These observations are extracted from ascii files and allows for loading
   of just about anything. Examples: 4D seismic, results from non ECLIPSE
   compatible simulators, etc.

 - :ref:`RFT_OBSERVATION <rft_observation>`: For loading RFT observations
   from a reservoir simulator RFT file. Examples are pressure and saturation
   values.


Please note that observations and datatypes are quite tightly linked together.
Before reading this you should have a firm grasp of the dynamic data types
as described in :ref:`Data types available in ERT <Data_types_available_in_ERT>`.


.. _summary_observation:

SUMMARY_OBSERVATION keyword
---------------------------

The keyword SUMMARY_OBSERVATION can be used to condition on any observation for
which the simulated value is in the :term:`summary files` with basename
:ref:`eclbase` produced by each :term:`realisation`, e.g. well rates, region properties, group and field rates etc.
A typical usage of SUMMARY_OBSERVATION is to condition on results from
separator tests.

In order to create a summary observation, four pieces of information
are needed: The observed value, the observation error, the time of
observation and a summary key. A typical summary observation is
created as follows:

.. code-block:: none

 SUMMARY_OBSERVATION SEP_TEST_2005
 {
    VALUE = 100.0;
    ERROR =     5;
    DATE  = 2005-08-21;
    KEY   = GOPR:BRENT;
 };

This will create an observation of group oil production for the Brent
group on 21th of august 2005. The observed value was 100 with a
standard deviation of 5. The name SEP_TEST_2005 will be used as a
label for the observation within ERT and must be unique.

Date format YYYY-MM-DD (ISO 8601) is required. Other time formats, like
DD/MM/YYYY or DD.MM.YYYY, are deprecated and their support will be removed in a
future release. The date format can also include hours and seconds:
"YYYY-MM-DDTHH:mm:ss". When the :ref:`eclbase` :term:`summary files` are read
from the realization, report times are rounded to seconds and matched to
closest observations with 1 second tolerance.

The item KEY is a :term:`summary key`, which is used to look up the simulated
value from the :term:`summary files` with basename :ref:`eclbase` from each
:term:`realisation`. To condition on the summary key VAR in a well, group or
region WGRNAME, use::

 KEY = VAR:WGRNAME;

For example, to condition on ``RPPW`` in region 8, use::

 KEY = RPPW:8;

Use the keyword ``RESTART`` to specify observation time as a restart number.
Use the keyword ``DAYS`` to specify observation time as the number of days relative
to the start of the simulation, where the start point is taken from the `REFCASE` or `TIMEMAP`.

Here are two examples:

.. code-block:: none

 -- Giving the observation time in terms of restart number.
 SUMMARY_OBSERVATION SEP_TEST_2005
 {
    VALUE    = 100;
    ERROR    =   5;
    RESTART  =  42;
    KEY      = GOPR:BRENT;
 };

 -- Giving the observation time in terms of days
 -- from simulation start.
 SUMMARY_OBSERVATION SEP_TEST_2008
 {
    VALUE    = 213;
    ERROR    =  10;
    DAYS     = 911;
    KEY      = GOPR:NESS;
 };

.. _error_modes:

Error modes for summary observations
------------------------------------

The item ERROR_MODE can take three different values: ABS, REL or RELMIN.
The default error mode for the :ref:`SUMMARY_OBSERVATION <summary_observation>`
keyword is ABS.

The default value for `ERROR_MIN` is 0.1.

ERT will not load an observation if the total error associated with an observation is zero.
A zero error is incompatible with the logic used in the history matching
process. Therefore, setting a minimum error is particularly important for
observations that could happen to be zero. For example, if an observation is the
water production rate and, at a given time, its value is zero, the relative
error will be zero, and the only error computed is the minimum error.

The error explicitizes the degree of uncertainty associated to the given
observation. It has an inverse effect on the weight that an observation
will have during the history matching process: the higher the error
specified for an observation, the smaller will be its weight during
the updating process. Therefore, it is important to have consistency
between setting up the errors and the degree of uncertainty in an
observation.

The default error mode and values can be changed as follows:

.. code-block:: none

  SUMMARY_OBSERVATION GOPR_FIELD_OBS_NAME
 {
    VALUE      = 0.9;
    DATE       = 2014-09-10;
    KEY        = GOPR:FIELD;
    ERROR       = 1000;
    ERROR_MODE  = ABS;
 };

This will set the observation error to an absolute value of 1000
for all observations of GOPR:FIELD.

Note that both the items ERROR and ERROR_MODE as well as
the whole definition shall end with a semi-colon.

If ERROR_MODE is set to REL, all observation errors will be set to the
observed values multiplied by ERROR. Thus, the following will
condition on water injection rate for the whole field with 20%
observation uncertainity:

.. code-block:: none

 SUMMARY_OBSERVATION GWIR_FIELD_OBS_NAME
 {
    VALUE      = 0.9;
    DATE       = 2014-09-10;
    KEY        = GWIR:FIELD;
    ERROR       = 0.20;
    ERROR_MODE  = REL;
 };

If you do not want the observation error to drop below a given
threshold, say 100, you can set ERROR_MODE to RELMIN and the
keyword ERROR_MIN:

.. code-block:: none

 SUMMARY_OBSERVATION GWIR_FIELD_OBS_NAME
 {
    VALUE      = 0.9;
    DATE       = 2014-09-10;
    KEY        = GWIR:FIELD;
    ERROR       = 0.20;
    ERROR_MODE  = RELMIN;
    ERROR_MIN   = 100;
 };

This error mode is also relevant for observations that may be zero,
for example water production rates.

Note that the configuration parser does not treat carriage return
different from space. Thus, the following statement is equivalent to
the previous:

.. _general_observation:

GENERAL_OBSERVATION keyword
---------------------------

The GENERAL_OBSERVATION keyword is used together with the GEN_DATA
type. This pair of observation and data types are typically
used when you want to update something special which does not fit into
any of the predefined types. Ert treats GENERAL_OBSERVATION (and also GEN_DATA)
as a list of numbers with no particular structure.
This is very flexible, but of course also a bit more complex to use:

.. code-block:: none

 GENERAL_OBSERVATION GEN_OBS1 {
    DATA     = SOME_FIELD;
    RESTART  = 20;
    OBS_FILE = some_file.txt;
 };

This example shows a minimum GENERAL_OBSERVATION. The keyword DATA
points to the GEN_DATA instance this observation is 'observing',
RESTART gives the report step when this observation is active.
OBS_FILE should be the name of a file with observation values,
and the corresponding uncertainties. The file with observations should
just be a plain text file with numbers in it, observations and
corresponding uncertainties interleaved.

An example of an ``OBS_FILE`` that defines three observations::

 1.46 0.26
 25.0 5.0
 5.00 1.00

In the example above it is assumed that the DATA
instance we are observing (i.e. comparing with) has the same number of
elements as the observation, i.e. three in this case. By using the
keyword INDEX_LIST you can select the elements of the
GEN_DATA instance you are interested in. Each index in INDEX_LIST
points to a line number in the GEN_DATA result file (which has one number per line).
Consider for example:

.. code-block:: none

   GENERAL_OBSERVATION GEN_OBS1 {
      DATA       = SOME_FIELD;
      INDEX_LIST = 0,3,9;
      RESTART    = 20;
      OBS_FILE   = some_file.txt;
   };

Here we use INDEX_LIST to indicate that we are interested in element
0, 3 and 9 of the GEN_DATA instance::

   GEN_DATA                     GEN_OBS1
   ========                     ===========
   1.56 <---------------------> 1.46  0.26
   23.0        /--------------> 25.0   5.00
   56.0        |    /---------> 5.00  1.00
   27.0 <------/    |           ===========
   0.2             |
   1.56             |
   1.78             |
   6.78             |
   9.00             |
   4.50 <-----------/
   ========


If ``INDEX_LIST`` not defined, Ert assumes that the observations point
to the first ``n`` ``GEN_DATA`` points:

.. code-block:: none

   GENERAL_OBSERVATION GEN_OBS1 {
      DATA       = SOME_FIELD;
      OBS_FILE   = some_file.txt;
   };

::

   GEN_DATA                     GEN_OBS1
   ========                     ===========
   1.56 <---------------------> 1.46  0.26
   23.0 <---------------------> 25.0   5.00
   56.0 <---------------------> 5.00  1.00
   27.0                         ===========
   0.2
   1.56
   1.78
   6.78
   9.00
   4.50
   ========


In addition to INDEX_LIST, it is possible to use INDEX_FILE which
points to a plain text file with indices, one value per line.
Finally, if your observation only has one value, you can
embed it in the config object with VALUE and ERROR.

Matching GEN_OBS and GEN_DATA
-----------------------------

It is important to match up the GEN_OBS observations with the
corresponding GEN_DATA simulation data correctly. If no ``REPORT_STEP``
and ``RESTART`` are provided to ``GEN_DATA`` and ``GENERAL_OBSERVATION``,
respectively, they will be given a default ``REPORT_STEP``
and ``RESTART`` of 0.

As a concrete example, the ert configuration file could include this line:

.. code-block:: none

   GEN_DATA RFT_BH67 RESULT_FILE:rft_BH67

While the observation configuration file could include this:

.. code-block:: none

   GENERAL_OBSERVATION GEN_OBS1 {
      DATA       = RFT_BH67;
      OBS_FILE   = some_file.txt;
   };

Before ERT starts we expect there to be a file called ``some_file.txt``  with the
observed values and the uncertainty. After the forward model has completed, ERT
will load the responses from a file called ``rft_BH67``.

If ``REPORT_STEP`` and ``RESTART`` are provided,
the ``GEN_DATA`` result files must have an embedded ``%d`` to indicate the
report step in them. To ensure that GEN_OBS and corresponding
GEN_DATA values match up correctly only the RESTART method is allowed
for GEN_OBS when specifying the time.
So consider a setup like this::

   -- Config file:
   GEN_DATA RFT_BH67 RESULT_FILE:rft_BH67_%d    REPORT_STEPS:20
   ...                                    /|\                /|\
   ...                                     |                  |
   -- Observation file:                    |                  |
   GENERAL_OBSERVATION GEN_OBS1 {          +------------------/
      DATA       = RFT_BH67;               |
      RESTART    = 20;   <-----------------/
      OBS_FILE   = some_file.txt;
   };

Here we see that the observation is active at report step 20, and we
expect the forward model to create a file rft_BH67_20 in each
realization directory.

.. _OPM Flow manual: https://opm-project.org/wp-content/uploads/2023/06/OPM_Flow_Reference_Manual_2023-04_Rev-0_Reduced.pdf


.. _rft_observation:

RFT_OBSERVATION keyword
-----------------------

The keyword RFT_OBSERVATION can be used to condition on observations for which simulated values are
in the RFT files with basename defined by :ref:`eclbase` produced by each :term:`realisation`,
e.g. pressure and saturation values.

A typical RFT observation is created as follows:

.. code-block:: none

   RFT_OBSERVATION rft_obs {
      WELL=PROD;
      DATE=2015-02-01;
      PROPERTY=PRESSURE;
      VALUE=3800;
      ERROR=10;
      TVD=8400;
      EAST=9500;
      NORTH=9500;
   };

This will create an observation of pressure in well PROD on 1st of February 2015.
The location of the measurement is given by the TVD, EAST and NORTH values,
where TVD is the true vertical depth below sea level, and EAST and NORTH are the coordinates
in the horizontal plane. The observed pressure value was 3800 with an observation error of 10.
Any property available in the RFT file can be used, e.g. PRESSURE, SWAT, SGAS, etc.
The error is given as an absolute value.

An RFT_OBSERVATION can alternatively be created by referring to an observation csv file containing
multiple observations.

In its simplest form this will look as follows:

.. code-block:: none

   RFT_OBSERVATION rft_obs {
      CSV=path/to/observation_file.csv;
   };

This will then default to look for PRESSURE values in the csv file.
If another property is required, this can be specified by adding the PROPERTY keyword, e.g:

.. code-block:: none

   RFT_OBSERVATION rft_obs {
      CSV=path/to/observation_file.csv;
      PROPERTY=SWAT;
   };

The CSV file needs to have the following columns as a minimum:
   - WELL_NAME
   - DATE
   - ERROR
   - NORTH
   - EAST
   - TVD

In addition the required property column needs to be present, e.g. PRESSURE (used by default) or SWAT, etc.

An optional ZONE column can be included to validate that observations are in the expected geological zones
when used with :ref:`ZONEMAP <zonemap>`.

An example of such a CSV could look like this:

.. code-block:: none

   "WELL_NAME", "DATE", "ZONE", "PRESSURE", "ERROR", "TVD", "NORTH", "EAST"
   "WELL1", "2013-03-31", "zone1", "3700", "10", "2000.0", "71.0", "30.0"
   "WELL1", "2013-04-30", "zone1", "3800", "10", "2000.0", "71.0", "30.0"
   "WELL2", "2014-03-31", "zone1", "3900", "10", "2000.0", "73.0", "33.0"

.. note::
   Such a CSV file can be produced by running fmu.tools.rms.create_rft_ertobs
   within RMS (https://equinor.github.io/fmu-tools/create_rft_ertobs.html). If
   given a CSV file with MD instead of TVD, NORTH, EAST coordinates the function
   will interpolate MD to TVD, NORTH, EAST along well paths, and will output
   a csv file compatible with ERT.


Using zones with RFT observations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When an RFT observation includes a ZONE identifier (either in the CSV file or specified directly),
ERT will validate that the measurement location falls within the expected geological zone defined
in the :ref:`ZONEMAP <zonemap>`. This provides an additional quality check to ensure observations
are correctly associated with reservoir zones.

If a zone is specified but no ZONEMAP is provided, or if the observation location doesn't match
the expected zone, the observation will be deactivated with a warning during the simulation.
