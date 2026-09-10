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

The format for the output files of `mock_sim2seis.py` can be set via the `--format` flag.
The default format is `.csv` and the supported formats are `.csv` and `.parquet`.
The following example sets the format to `.parquet`.
````
python mock_sim2seis.py --format parquet
````

Run setup:
```
ert ensemble_experiment sim2seis.ert
```
Note that setup expects observation data in both `.csv` and `.parquet` formats.

Observation data is found under `share/preprocessed/tables`. Modelled data is found
under `share/results/tables`. Note that modelled data have the same structure as
observation data, but `OBS` column is actually a `VALUE` column and `OBS_ERROR` should
be ignored.

### Drogon example
While the original mock setup was created to minimize data and keep full control of the inputs,
a more realistic case with a larger number of data points appeared to be useful too.

`drogon` directory contains such files produced by `fmu-sim2seis` itself.

Observation files in `obs` directory are:
- `topvolantis--amplitude_full_mean_depth--20190701_20180101.csv`
- `topvolantis--amplitude_full_min_depth--20190701_20180101.csv`

`fake-realization-responses` directory contains modelled data. `fmu-sim2seis` produces
just one realization per setup, but can do it for many dates. In this setup those files
for different dates are used as if they were responses for different realizations. This
is done to create diversity between realizations. There are in total 3 realizations
available and iterations return data files for them in a different order.

Note that because modelled responses do not depend on parameters chosen by ERT, update
step will not lead to responses converging to observations.

Run setup:
```
ert ensemble_experiment sim2seis_drogon.ert
```

### Assumptions
Assumptions in the script are taken from analyzing results of the fmu-sim2seis test-data
runs.
 - All data is assumed to have the same UTM coordinates in the same order.
 - Same UTM coordinates always belong to the same region.
 - Observation errors can vary from row to row.
 - Number of lines in each file (observation and modelled) is always the same.
