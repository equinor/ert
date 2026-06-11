## Seismic example
Used to test [fmu-sim2seis setup](https://github.com/equinor/fmu-sim2seis/tree/main).

### Why mock
Creating a real setup is challenging:
 - the only [data setup](https://github.com/equinor/fmu-sim2seis/tree/main/tests/data)
   by domain experts that we have is based on fmu-drogon
 - files are sizeable and grid is large
 - fmu-sim2seis setup uses files produced by other models and there is only one version
   of them. We, on the other hand, want to have a different setup for each realization.
 - dependencies between data are complex, so mocking the input data for a real run is
   beyond typical developer knowledge and can lead to errors due to unexpected
   dependencies.

So instead we are using a black-box forward model accepting some parameters and
producing output files of similar kind to fmu-sim2seis.

Current setup might be too simple, so it is a subject to change as we go forward. It
should also be checked from time to time to be compliant with fmu-sim2seis.

### Run

To be able to execute the setup file `mock_sim2seis.py` must be set as executable.

If needed, run the following to regenerate observation data:

```
python mock_sim2seis.py --observations
```

Run setup:
```
ert ensemble_experiment sim2seis.ert
```

Observation data is found under `share/preprocessed/tables`. Modelled data is found
under `share/results/tables`. Note that modelled data have the same structure as
observation data, but `OBS` column is actually a `VALUE` column and `OBS_ERROR` should
be ignored.


### Assumptions
Assumptions in the script are taken from analyzing results of the fmu-sim2seis test-data
runs.
 - All data is assumed to have the same UTM coordinates in the same order.
 - Same UTM coordinates always belong to the same region.
 - Observation errors can vary from row to row.
 - Number of lines in each file (observation and modelled) is always the same.
