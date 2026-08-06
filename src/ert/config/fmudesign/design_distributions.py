"""Module for random sampling of parameter values from distributions."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import probabilit
from scipy import stats


def validate_params(distname: str, parameters: list[str]) -> list[float]:
    """Common parameter validation for all distributions.

    Example:
    >>> validate_params('normal', ['0', '-3.14', '1e10'])
    [0.0, -3.14, 10000000000.0]
    >>> validate_params('normal', ['inf'])
    Traceback (most recent call last):
        ...
    ValueError: Parameter 1 in distribution normal is not finite: inf
    """
    new_parameters: list[float] = []

    for i, parameter in enumerate(parameters):
        try:
            new_parameters.append(float(parameter))
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Parameter {i + 1} in distribution {distname} not "
                f"convertible to number: {parameter}"
            ) from e

        if not np.isfinite(new_parameters[i]):
            raise ValueError(
                f"Parameter {i + 1} in distribution {distname} "
                f"is not finite: {parameter}"
            )

    return new_parameters


def quantiles_to_values(
    *,
    quantiles: npt.NDArray[Any],
    values: npt.NDArray[Any],
    probabilities: npt.NDArray[Any] | None = None,
) -> npt.NDArray[Any]:
    """Maps quantiles to values (which can be categorical or not).

    Assume values = [A, B, C], then probabilities = [1, 1, 1] = [1/3, 1/3, 1/3],
    first we bin the interval [0, 1) into segments matching the probabilities:
    - A: [0, 1/3)
    - B: [1/3, 2/3)
    - C: [2/3, 1)
    Then we map the quantiles into the ranges back onto the original values.

    Examples
    --------
    >>> values = np.array(["A", "B", "C"])
    >>> quantiles = np.array([0, 1/3, 0.5, 0.8])
    >>> quantiles_to_values(quantiles=quantiles, values=values)
    array(['A', 'B', 'B', 'C'], dtype='<U1')

    >>> probabilities = np.array([0.2, 0.3, 0.5])
    >>> quantiles = (np.arange(1, 10)) / 10
    >>> quantiles_to_values(quantiles=quantiles, values=values,
    ...                     probabilities=probabilities) # [0.1, 0.2, ...]
    array(['A', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'C'], dtype='<U1')
    """
    quantiles = np.array(quantiles)
    values = np.array(values)

    # If no probabilities are given, assume equal
    if probabilities is None:
        probabilities = np.ones(len(values)) / len(values)

    if not np.isclose(np.sum(probabilities), 1.0):
        probabilities /= np.sum(probabilities)

    assert np.all(probabilities >= 0)

    # Create bin edges
    edges = np.cumsum([0, *list(probabilities)])
    # Map to bin indices, then back onto the original values
    bin_indices = np.digitize(quantiles, edges, right=False) - 1
    return values[bin_indices]


def to_probabilit(
    distname: str,
    dist_parameters: Sequence[str],
) -> probabilit.modeling.AbstractDistribution:
    """
    Prepare scipy distributions with parameters
    Args:
        distname (str): distribution name 'normal', 'lognormal', 'triang',
        'uniform', 'logunif', 'discrete', 'pert', 'beta'
        dist_parameters (list): list with parameters for distribution
    Returns:
        array with sampled values
    """

    distname = distname.lower().strip()

    # A discrete variable is a distribution over categoricals, e.g. ('A', 'B', 'C')
    # with weights (0.5, 0.3, 0.2). The way we deal with them is that we sample uniform
    # values, then assign the interval [0, 0.5) to A, [0.5, 0.8) to B and [0.8, 1) to C.
    # This means that we can "correlate" these variables, in the sense that if their
    # underlying Uniforms are correlated, then the categorical values will often match
    # too. To accomplish all of this we assign _values and _probabilities to the
    # distribution instances below. This "correlation" only exists in a narrow specific
    # sense of course.

    if distname.startswith("disc"):
        if len(dist_parameters) == 1:
            values_str = str(dist_parameters[0])
            values = [v.strip() for v in values_str.split(",")]
            distr = probabilit.Distribution("uniform")
            distr._values = np.array(values)
            return distr
        values_str, probabilities_str = map(str, dist_parameters)
        values = [v.strip() for v in values_str.split(",")]
        probabilities = [float(v.strip()) for v in probabilities_str.split(",")]
        if len(values) != len(probabilities):
            raise ValueError(
                "Length mismatch for discrete distribution, "
                f"dist_param1 has length {len(values)}, "
                f"but dist_param2 has length {len(probabilities)}. "
                "dist_param1 and dist_param2 must have the same numer of "
                "entries for discrete distributions."
            )
        distr = probabilit.Distribution("uniform")
        distr._values = np.array(values)
        distr._probabilities = np.array(probabilities)
        return distr

    # Special case for constant
    if distname.startswith("const"):
        return probabilit.Constant(dist_parameters[0])

    # Convert parameters
    parameters: list[float] = validate_params(
        distname=distname, parameters=list(dist_parameters)
    )

    match (distname, parameters):
        # ================== NORMAL ==================

        case [_, (p10, p90)] if distname.startswith("normal_p10_p90"):
            # We use the equations
            # p10 = mu - z*sigma and p90 = mu + z*sigma
            # to find mu and sigma
            z_score = stats.norm.ppf(0.9)
            mean = (p10 + p90) / 2
            std = (p90 - p10) / (2 * z_score)
            return probabilit.distributions.Normal(mean=mean, std=std)

        case [_, (p10, p90, low, high)] if distname.startswith("normal_p10_p90"):
            z_score = stats.norm.ppf(0.9)
            mean = (p10 + p90) / 2
            std = (p90 - p10) / (2 * z_score)
            return probabilit.distributions.TruncatedNormal(
                mean=mean, std=std, low=low, high=high
            )

        case [_, (mean, std)] if distname.startswith("norm"):
            return probabilit.distributions.Normal(mean=mean, std=std)

        case [_, (mean, std, low, high)] if distname.startswith("norm"):
            return probabilit.distributions.TruncatedNormal(
                mean=mean, std=std, low=low, high=high
            )

        case [_, parameters] if distname.startswith("norm"):
            raise ValueError(
                f"Normal must have 2 or 4 parameters, got: "
                f"{len(parameters)} ({parameters})"
            )

        # ================== LOGNORMAL ==================

        case [_, (mu, sigma)] if distname.startswith("logn"):
            return probabilit.distributions.Lognormal.from_log_params(
                mu=mu, sigma=sigma
            )

        case [_, (mu, sigma, low, high)] if distname.startswith("logn"):
            # (mu, sigma) are defined in log-space, but (low, high)
            # are defined on exp-space
            return probabilit.modeling.Exp(
                probabilit.distributions.TruncatedNormal(
                    mu, sigma, low=np.log(low), high=np.log(high)
                )
            )

        case [_, parameters] if distname.startswith("logn"):
            raise ValueError(
                f"Lognormal must have 2 or 4 parameters, got: "
                f"{len(parameters)} ({parameters})"
            )

        # ================== UNIFORM ==================

        case [_, (p10, p90)] if distname.startswith("uniform_p10_p90"):
            length = (p90 - p10) / 0.8
            minimum = p10 - 0.1 * length
            maximum = p90 + 0.1 * length
            return probabilit.distributions.Uniform(minimum=minimum, maximum=maximum)

        case [_, (minimum, maximum)] if distname.startswith("unif"):
            return probabilit.distributions.Uniform(minimum=minimum, maximum=maximum)

        case [_, parameters] if distname.startswith("unif"):
            raise ValueError(
                f"Uniform must have 2 parameters, got: {len(parameters)} ({parameters})"
            )

        # ================== TRIANGULAR ==================

        case [_, (low, mode, high)] if distname.startswith("triangular_p10_p90"):
            return probabilit.distributions.Triangular(
                low=low, mode=mode, high=high, low_perc=0.1, high_perc=0.9
            )

        case [_, (minimum, mode, maximum)] if distname.startswith("triang"):
            return probabilit.distributions.Triangular(
                low=minimum, mode=mode, high=maximum, low_perc=0.0, high_perc=1.0
            )

        case [_, parameters] if distname.startswith("triang"):
            raise ValueError(
                f"Triangular must have 3 parameters, got: "
                f"{len(parameters)} ({parameters})"
            )

        # ================== BETA ==================

        case [_, (a, b)] if distname.startswith("beta"):
            # Defaults to probabilit.Distribution("beta", a=a, b=b, loc=0, scale=1)
            return probabilit.Distribution("beta", a=a, b=b)

        case [_, (a, b, low, high)] if distname.startswith("beta"):
            loc = low
            scale = high - low
            return probabilit.Distribution("beta", a=a, b=b, loc=loc, scale=scale)

        case [_, parameters] if distname.startswith("beta"):
            raise ValueError(
                f"Beta must have 2 or 4 parameters, got: "
                f"{len(parameters)} ({parameters})"
            )

        # ================== PERT ==================

        case [_, (low, mode, high)] if distname.startswith("pert_p10_p90"):
            return probabilit.distributions.PERT(
                low=low, mode=mode, high=high, low_perc=0.1, high_perc=0.9
            )

        case [_, (low, mode, high, scale)] if distname.startswith("pert_p10_p90"):
            return probabilit.distributions.PERT(
                low=low, mode=mode, high=high, low_perc=0.1, high_perc=0.9, gamma=scale
            )

        case [_, (minimum, mode, maximum)] if distname.startswith("pert"):
            return probabilit.distributions.PERT(
                low=minimum, mode=mode, high=maximum, low_perc=0.0, high_perc=1.0
            )

        case [_, (minimum, mode, maximum, scale)] if distname.startswith("pert"):
            return probabilit.distributions.PERT(
                low=minimum,
                mode=mode,
                high=maximum,
                low_perc=0.0,
                high_perc=1.0,
                gamma=scale,
            )

        case [_, parameters] if distname.startswith("pert"):
            raise ValueError(
                f"PERT must have 3 or 4 parameters, got: "
                f"{len(parameters)} ({parameters})"
            )

        # ================== LOGUNIFORM ==================

        case [_, (low, high)] if distname.startswith("logunif"):
            return probabilit.Distribution("loguniform", low, high)

        case [_, parameters] if distname.startswith("logunif"):
            raise ValueError(
                f"Loguniform must have 2 parameters, got: "
                f"{len(parameters)} ({parameters})"
            )

        case [distname, parameters]:
            raise ValueError(f"Invalid combination of {distname=} and {parameters=}.")


def is_number(teststring: str) -> bool:
    """Test if a string can be parsed as a float"""
    try:
        return not np.isnan(float(teststring))
    except ValueError:
        return False


def read_correlations(excel_filename: str, corr_sheet: str) -> pd.DataFrame:
    """Read a correlation matrix from an Excel sheet.

    The sheet must have rows/columns with variable names. They must match.
    The upper-triangular part must be empty strings. The lower triangular part
    must be specified.

    Args:
        excel_filename (str): name of Excel file containing correlation matrix
        corr_sheet (str): name of sheet containing correlation matrix

    Returns:
        pd.DataFrame: Dataframe with correlations, parameter names
            as column and index
    """
    if not str(excel_filename).endswith(".xlsx"):
        raise ValueError(
            "Correlation matrix filename should be on Excel format and end with .xlsx"
        )

    correlations = (
        pd.read_excel(
            excel_filename,
            sheet_name=corr_sheet,
            index_col=0,
            # A user reported failures when a single space ' ' was present
            # in the upper triangular part. Therefore we add spaces as NaN too.
            na_values=[" " * i for i in range(10)],
            engine="openpyxl",
        )
        .dropna(axis=0, how="all")
        # Remove any 'Unnamed' columns that Excel/pandas may have automatically added.
        .loc[:, lambda df: ~df.columns.str.contains("^Unnamed")]
        # Remove whitespace
        .rename(columns=str.strip)
        .rename(index=str.strip)
    )

    if list(correlations.index) != list(correlations.columns):
        msg = (
            "Mismatch between column and index in correlation "
            f"matrix sheet: {corr_sheet!r}\n"
            f"Column: {correlations.columns.tolist()}\n"
            f"Index : {correlations.index.tolist()}\n"
            "These values must match exactly. Please fix sheet "
            f"{corr_sheet!r} in file {excel_filename!r}."
        )
        raise ValueError(msg)

    arr = correlations.to_numpy(copy=True)
    upper_idx = np.triu_indices_from(arr, k=1)
    lower_idx = np.tril_indices_from(arr, k=0)  # Include diag
    lower_entries = arr[lower_idx]

    if not np.all(np.isnan(arr[upper_idx])):
        raise ValueError(
            "All upper-triangular elements in matrix in "
            f"corr sheet {corr_sheet} must be blank."
        )

    if not np.all(np.isfinite(lower_entries)):
        raise ValueError(
            "All lower-triangular elements in matrix in "
            f"corr sheet {corr_sheet} must be specified."
        )

    if np.any((lower_entries < -1) | (lower_entries > 1)):
        raise ValueError(
            "All lower-triangular elements in matrix in "
            f"corr sheet {corr_sheet} must be between -1 and 1."
        )

    # Build symmetric matrix from lower triangle
    np.nan_to_num(arr, copy=False, nan=0.0)
    mat = arr + arr.T
    np.fill_diagonal(mat, 1.0)
    return pd.DataFrame(mat, index=correlations.index, columns=correlations.columns)
