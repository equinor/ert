from typing import Any

import numpy as np
import pandas as pd


class QualityReporter:
    """This class is responsible for quality reporting a dataframe with samples.
    It has methods to print statistical outputs.

    Examples
    >>> import pandas as pd
    >>> df = pd.DataFrame({"a": [2, 4, 2, 3, 4, 3, 2, 3, 4, 5],
    ...                    "b": [2, 4, 2, 4, 2, 4, 2, 3, 2, 3],
    ...                    "c": list("asdfsdfsdf")})
    >>> variables = {"a": ["Normal",[0, 1]], "b": ["Expon",[1]], "c":["Discrete"]}
    >>> quality_reporter = QualityReporter(df, variables=variables)
    >>> quality_reporter.print_numeric()
    ================ CONTINUOUS PARAMETERS ================
       mean       std  min  10%  50%  90%  max
    a   3.2  1.032796  2.0  2.0  3.0  4.1  5.0
    b   2.8  0.918937  2.0  2.0  2.5  4.0  4.0
    >>> quality_reporter.print_discrete()
    ================ DISCRETE PARAMETERS ================
    | c   |   proportion |
    |:----|-------------:|
    | s   |          0.3 |
    | d   |          0.3 |
    | f   |          0.3 |
    | a   |          0.1 |
    """

    def __init__(self, df: pd.DataFrame, variables: dict[str, list[Any]]) -> None:
        """Initialize QualityReporter with dataframe and variable descriptions.

        Args:
            df: DataFrame containing the samples
            variables: Dictionary mapping variable names to their
                distribution descriptions
                      e.g., {"COST": ["normal", [0.0, 1.0]]}
        """
        self.df: pd.DataFrame = df.loc[:, list(variables.keys())]
        self.variables = variables
        assert not self.df.empty

    def print_numeric(self) -> None:
        """Print statistics for all numerical columns."""
        df_numeric = self.df.select_dtypes(include="number")
        if df_numeric.empty:
            return

        print("=" * 16, "CONTINUOUS PARAMETERS", "=" * 16)

        with pd.option_context(
            "display.max_rows", None, "display.max_columns", None, "display.width", None
        ):
            print(
                df_numeric.describe(percentiles=[0.1, 0.5, 0.9]).T.drop(
                    columns=["count"]
                )
            )

    def print_discrete(self) -> None:
        """Print statistics for all discrete (non-numerical) columns."""
        df_non_numeric = self.df.select_dtypes(exclude="number")
        if df_non_numeric.empty:
            return

        print("=" * 16, "DISCRETE PARAMETERS", "=" * 16)

        for column in df_non_numeric.columns:
            print(self.df[column].value_counts(normalize=True).round(3).to_markdown())

    def print_correlation(self, corr_name: str, df_corr: pd.DataFrame) -> None:
        """Print information about desired and achieved correlation matrices.

        Args:
            corr_name: Name of the correlation group
            df_corr: DataFrame containing the desired correlation matrix, which
                must be a valid correlation matrix
        """
        corr = self.df[df_corr.columns].select_dtypes(include="number").corr()
        # Do not print if only a single variable remains
        if corr.shape[1] == 1:
            return

        print(
            "=" * 16,
            f"CORRELATION_GROUP {corr_name!r} (num variables: {len(df_corr.columns)})",
            "=" * 16,
        )

        if len(df_corr) <= 12:
            print("Desired correlation:")
            print_corrmat(df_corr)
        else:
            print("Skipping printing desired correlation. Matrix too large.")

        # No correlation in samples (e.g. discrete variables)
        if corr.empty:
            return

        if len(corr) <= 12:
            print("Observed (Pearson) correlation in samples:")
            print_corrmat(corr)
        else:
            print("Skipping printing observed (Pearson) correlation. Matrix too large.")

        target = df_corr.loc[corr.index, corr.columns].to_numpy()
        idx_low_triang = np.tril_indices_from(target, k=-1)
        diffs = corr.to_numpy()[idx_low_triang] - target[idx_low_triang]
        corr_rmse = np.sqrt(np.mean(diffs**2))
        print(
            "Distance metrics between target correlation matrix "
            "and empirical correlation matrix"
        )
        print(f" - Root Mean Squared Error    (RMSE): {corr_rmse:.6f}")

        if corr_rmse > 0.05:
            print(
                "Target correlation matrix and empirical correlation achieved",
                " in data does not match well\n"
                "This is natural with few samples, or very high/low desired correlations, "  # ruff: ignore[line-too-long]
                "or distributions that are far from\nnormal (e.g. lognormal)."
                " Setting 'correlation_iterations' to 999 in the general input sheet might help.",  # ruff: ignore[line-too-long]
            )


def print_corrmat(df_corrmat: pd.DataFrame) -> None:
    """Print a correlation matrix.

    Example:
    >>> values = np.array([[  1, -0,  0.9],
    ...                    [ -0,  1,    0],
    ...                    [0.9,  0,    1]])
    >>> vars_ = ['OWC1', 'OWC2', 'OWC3']
    >>> df_corrmat = pd.DataFrame(values, index=vars_, columns=vars_)
    >>> print_corrmat(df_corrmat)
    |          |   (1) |   (2) |   (3) |
    |:---------|------:|------:|------:|
    | (1) OWC1 |  1.00 |       |       |
    | (2) OWC2 |  0.00 |  1.00 |       |
    | (3) OWC3 |  0.90 |  0.00 |  1.00 |
    """
    values = df_corrmat.to_numpy(copy=True)
    assert np.allclose(values, values.T)
    # Make slightly negative values positive
    mask = np.isclose(values, 0)
    values[mask] = np.abs(values[mask])
    df_corrmat = pd.DataFrame(
        values, index=df_corrmat.index, columns=df_corrmat.columns
    )

    # Compress columns into integers so we can show more on the screen
    assert list(df_corrmat.columns) == list(df_corrmat.index)
    varnames = list(df_corrmat.columns)
    df_corrmat = df_corrmat.set_axis(
        [f"({i})" for i, _ in enumerate(varnames, 1)], axis=1
    )
    df_corrmat = df_corrmat.set_axis(
        [f"({i}) {varname}" for i, varname in enumerate(varnames, 1)], axis=0
    )

    # Remove upper triangular part for prettier printing
    def formatter(x: float) -> str:
        return np.format_float_positional(x, precision=2, unique=True, min_digits=2)

    mask = np.triu(np.ones_like(df_corrmat, dtype=bool), k=1)
    df_display = df_corrmat.astype(float).map(formatter)
    df_display[mask] = ""
    print(
        df_display.to_markdown(
            floatfmt=".2f",
            disable_numparse=True,
            numalign="right",
            stralign="right",
            colalign=("left",),
        )
    )
