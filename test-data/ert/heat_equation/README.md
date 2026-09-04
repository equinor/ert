# Estimate heat transfer coefficients of a plate using the heat equation as forward model

Based on the following tutorial: https://github.com/equinor/dass/blob/main/notebooks/ES_2D_Heat_Equation.ipynb

- **generate_files.py:** File run once to generate `CASE.EGRID` and observations.
- **heat_equation.py:** Forward model implementing the heat eqauation.

## Observations

`observations_loc.txt` contains temperature observations at four
different locations (grid coordinates) in the plate:
`(10.5, 25.5)`, `(25.5, 10.5)`, `(40.5, 25.5)`, and `(25.5, 40.5)`.

Each location is observed at several simulation times. The
observations are added as `SUMMARY_OBSERVATION` with a localization radius
of 20 grid cells.

## Distance based localization

The observation `LOCALIZATION` block provides the metadata required for
distance based localization. To use it for the conductivity field parameter, set the
field strategy in `config.ert` to:

```
ANALYSIS_SET_VAR PARAMETERS FIELD DISTANCE
```

Please note, the current `config.ert` currently uses adaptive localization;
switching the field strategy to `DISTANCE` activates distance based localization.
