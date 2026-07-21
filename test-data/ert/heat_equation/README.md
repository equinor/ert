# Estimate heat transfer coefficients of a plate using the heat equation as forward model

Based on the following tutorial: https://github.com/equinor/dass/blob/main/notebooks/ES_2D_Heat_Equation.ipynb

- **generate_files.py:** File run once to generate `CASE.EGRID`, `HEAT.SMSPEC`, and observations. `HEAT.SMSPEC` is staged into each runpath with `COPY_FILE`.
- **heat_equation.py:** Forward model implementing the heat eqauation.
