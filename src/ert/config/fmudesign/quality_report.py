import copy
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as sp
import seaborn as sns
from probabilit.correlation import (
    nearest_correlation_matrix,
)

from semeio.fmudesign.design_distributions import to_probabilit

if TYPE_CHECKING:
    from matplotlib.patches import Rectangle

COLORS = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])


class QualityReporter:
    """This class is responsible for quality reporting a dataframe with samples.
    It has methods to print statistical outputs and save figures to disk.

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
        self.variables: dict[str, list[Any]] = copy.deepcopy(variables)
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

    @staticmethod
    def _create_output_dir(output_dir: Path | None) -> Path | None:
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            return output_dir
        return None

    def plot_columns(self, output_dir: Path | None = None) -> None:
        """Loop through all columns and plot them, saving to disk if
        `output_dir` is given and exists.

        Args:
            output_dir: Optional directory path to save plots. If None, plots
                       are not saved to disk.
        """
        df_numeric = self.df.select_dtypes(include="number")
        df_non_numeric = self.df.select_dtypes(exclude="number")

        print("=" * 16, "GENERATING VARIABLE PLOTS", "=" * 16)

        # Create output directory if specified
        output_path = self._create_output_dir(output_dir)

        # Plot numeric columns
        for column in df_numeric.columns:
            # We do not plot constant distributions
            if self.variables[column][0].lower().startswith("const"):
                continue

            fig, _ax = self.plot_numeric(
                series=self.df[column],
                var_name=column,
                var_description=self.variables[column],
            )

            if output_path is not None:
                filename = output_path / f"{column}.png"
                fig.savefig(filename, dpi=200)
                print(f" - Saved variable: {filename}")

            plt.close(fig)

        # Plot discrete columns
        for column in df_non_numeric.columns:
            fig, _ax = self.plot_discrete(
                series=self.df[column],
                var_name=column,
                var_description=self.variables[column],
            )

            if output_path is not None:
                filename = output_path / f"{column}.png"
                fig.savefig(filename, dpi=200)
                print(f" - Saved file: {filename}")

            plt.close(fig)

    @staticmethod
    def plot_numeric(
        series: pd.Series, var_name: str, var_description: list[Any]
    ) -> tuple[plt.Figure, plt.Axes]:
        """Create a plot for a single numeric column, returning (fig, ax).

        Args:
            series: Pandas series containing the numeric data
            var_name: Name of the variable
            var_description: Description of the variable distribution

        Returns:
            Tuple of (matplotlib Figure, matplotlib Axes)
        """
        fig, ax = plt.subplots(figsize=(6, 4))
        bins = max(int(math.sqrt(len(series))), 10)
        sns.histplot(data=series, stat="density", bins=bins, ax=ax)
        sns.kdeplot(series, ax=ax, color="blue", label="KDE")

        # Create string to describe distiribution
        var_string = (
            f"{var_description[0]}~("
            + ", ".join(
                f"dist_param{i + 1}={v}" for i, v in enumerate(var_description[1])
            )
            + ")"
        )

        ax.set_title(f"{var_name}\n{var_string}", fontsize=7)

        # Add plot of expected PDF
        dist_name, dist_parameters = var_description[:2]

        try:
            dist = to_probabilit(dist_name, dist_parameters)
            x = np.linspace(series.min(), series.max(), 1000)
            pdf = dist.to_scipy().pdf(x)
            ax.plot(x, pdf, color="red", lw=2, ls="--", label="expected PDF")
        except AttributeError:
            print(f" - Plot warning: PDF not plotted for {var_name}")

        # Add rugplot
        ax.scatter(
            series.to_numpy(),
            np.zeros(len(series)),
            marker="|",
            color=COLORS[1],
            alpha=0.8,
        )

        # Add average and quantiles to the plot
        mean = series.mean()
        ax.axvline(
            x=mean,
            color="black",
            ls="-",
            alpha=0.8,
            label=f"mean={mean:.2e}",
        )

        quantiles = [0.1, 0.5, 0.9]
        for q in quantiles:
            quantile_value = series.quantile(q=q)
            P_label = f"{q * 100:.0f}".zfill(2)  # e.g. 0.05 => '05'
            ax.axvline(
                x=quantile_value,
                color="black",
                ls="--",
                alpha=0.8,
                label=f"P{P_label}={quantile_value:.2e}",
            )

        ax.grid(visible=True, ls="--", alpha=0.5)
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=7)
        fig.tight_layout()

        return fig, ax

    @staticmethod
    def plot_discrete(
        series: pd.Series, var_name: str, var_description: list[Any]
    ) -> tuple[plt.Figure, plt.Axes]:
        """Create a plot for a single discrete column, returning (fig, ax).

        Args:
            series: Pandas series containing the discrete data
            var_name: Name of the variable
            var_description: Description of the variable distribution

        Returns:
            Tuple of (matplotlib Figure, matplotlib Axes)
        """
        fig, ax = plt.subplots(figsize=(6, 4))

        # Calculate normalized proportions
        proportions = series.value_counts(normalize=True)
        value_counts = series.value_counts(normalize=False)

        # Create DataFrame for seaborn
        plot_data = pd.DataFrame(
            {var_name: proportions.index, "proportion": proportions.to_numpy()}
        ).sort_values(by=var_name)

        # Use seaborn barplot with normalized values
        sns.barplot(data=plot_data, x=var_name, y="proportion", ax=ax)

        # Create string to describe distiribution
        var_string = (
            f"{var_description[0]}~("
            + ", ".join(
                f"dist_param{i + 1}={v}" for i, v in enumerate(var_description[1])
            )
            + ")"
        )

        ax.set_title(f"{var_name}\n{var_string}", fontsize=7)
        ax.set_ylabel("Proportion")

        # Add percentage labels on bars
        for _proportion, count, p in zip(
            proportions, value_counts, ax.patches, strict=False
        ):
            rect = cast("Rectangle", p)
            # assert math.isclose(_proportion, rect.get_height())
            percentage = f"{rect.get_height():.1%} (n={count:.0f})"
            ax.annotate(
                percentage,
                (rect.get_x() + rect.get_width() / 2.0, rect.get_height()),
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.grid(visible=True, ls="--", alpha=0.5, axis="y")
        ax.tick_params(axis="x", rotation=0)
        fig.tight_layout()

        return fig, ax

    def print_correlation(self, corr_name: str, df_corr: pd.DataFrame) -> None:
        """Print information about desired and achieved correlation matrices.

        Args:
            corr_name: Name of the correlation group
            df_corr: DataFrame containing the desired correlation matrix
        """
        corr_arr = df_corr.to_numpy()
        assert np.allclose(corr_arr, corr_arr.T)
        assert df_corr.shape[0] == df_corr.shape[1]

        # Get lower triangular indices
        idx_low_triang = np.tril_indices_from(corr_arr, k=-1)
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

        nearest_corr = nearest_correlation_matrix(
            corr_arr, weights=None, eps=1e-6, verbose=False
        )
        diffs = nearest_corr[idx_low_triang] - corr_arr[idx_low_triang]
        corr_rmse = np.sqrt(np.mean(diffs**2))

        if corr_rmse > 1e-2:
            print(f"The desired correlation matrix is not valid => {corr_rmse=:.2f}")
            print("Closest valid correlation matrix (used as target):")
            df_nearest_corr = pd.DataFrame(
                nearest_corr, columns=df_corr.columns, index=df_corr.index
            )
            print_corrmat(df_nearest_corr)
        else:
            print("The desired correlation matrix is valid")

        # No correlation in samples (e.g. discrete variables)
        if corr.empty:
            return

        if len(corr) <= 12:
            print("Observed (Pearson) correlation in samples:")
            print_corrmat(corr)
        else:
            print("Skipping printing observed (Pearson) correlation. Matrix too large.")

        # Difference between achieved corr in samples and target
        diffs = corr.to_numpy()[idx_low_triang] - nearest_corr[idx_low_triang]
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
                "This is natural with few samples, or very high/low desired correlations, "  # noqa: E501
                "or distributions that are far from\nnormal (e.g. lognormal)."
                " Setting 'correlation_iterations' to 999 in the general input sheet might help.",  # noqa: E501
            )

    def plot_correlation(
        self,
        corr_name: str,
        df_corr: pd.DataFrame,
        *,
        output_dir: Path | None = None,
        show: bool,
    ) -> None:
        """Plot correlation group of variables.

        Args:
            corr_name: Name of the correlation group
            df_corr: DataFrame containing the correlation structure to plot
            output_dir: Optional directory path to save plots
            show: Whether or not to show the matplotlib figure
        """
        # Short circuit this case, as there is nothing to do
        if (not show) and (output_dir is None):
            return

        def corrfunc(
            x: npt.NDArray[np.float64],
            y: npt.NDArray[np.float64],
            **kwargs: Any,  # noqa: ANN401
        ) -> None:
            # Add correlations and grid to plots
            r, _ = sp.stats.pearsonr(x, y)
            ax = plt.gca()
            ax.annotate(
                r"$\rho=$" + f"{r:.2f}",
                xy=(0.05, 0.95),
                xycoords=ax.transAxes,
            )
            ax.grid(visible=True, ls="--", alpha=0.5)

        def add_grid(*args: Any, **kwargs: Any) -> None:  # noqa: ANN401
            ax = plt.gca()
            ax.grid(visible=True, ls="--", alpha=0.5)

        df = self.df[df_corr.columns].select_dtypes(include="number")

        # Only plot if two or more variables remain
        if df.shape[1] <= 1:
            return

        pairgrid = sns.PairGrid(df)

        pairgrid.map_upper(sns.kdeplot)
        pairgrid.map_upper(add_grid)
        bins = max(int(math.sqrt(len(df))), 30)
        pairgrid.map_diag(sns.histplot, bins=bins, kde=True)

        pairgrid.map_lower(sns.scatterplot, s=10, alpha=0.6)
        pairgrid.map_lower(corrfunc)

        # Add rugplots to diagonal plots
        for i, var in enumerate(df.columns):
            pairgrid.diag_axes[i].scatter(  # type: ignore[index]
                df[var].to_numpy(),
                np.zeros(len(df)),
                marker="|",
                color="black",
                alpha=0.5,
            )

        output_path = self._create_output_dir(output_dir)
        if output_path is not None:
            filename = output_path / f"{corr_name}.png"
            pairgrid.savefig(filename, dpi=200)
            print(f" - Saved correlation: {filename}")

        if show:
            plt.show()

        plt.close(pairgrid.fig)

    def plot_correlation_heatmap(
        self,
        corr_name: str,
        df_corr: pd.DataFrame,
        *,
        output_dir: Path | None = None,
        show: bool,
    ) -> None:
        """Plot correlation heapmap of group of variables.

        Args:
            corr_name: Name of the correlation group
            df_corr: DataFrame containing the correlation structure to plot
            output_dir: Optional directory path to save plots
            show: Whether or not to show the matplotlib figure
        """
        # Short circuit this case, as there is nothing to do
        if (not show) and (output_dir is None):
            return

        df = self.df[df_corr.columns].select_dtypes(include="number")

        # Only plot if two or more variables remain
        if df.shape[1] <= 1:
            return

        correlation_matrix = df.corr()
        n_vars = len(correlation_matrix)

        # Roughly try to create a figure of a good size
        size = 3 + n_vars * 0.15
        fig, ax = plt.subplots(1, 1, figsize=(size, size))

        # Custom annotation
        arr = correlation_matrix.to_numpy()
        annot_arr = np.empty_like(arr, dtype=object)
        for (i, j), value in np.ndenumerate(arr):
            annot_arr[i, j] = "1" if i == j else f"{value:.2f}".replace("0.", ".")
        annot_matrix = pd.DataFrame(
            annot_arr,
            index=correlation_matrix.index,
            columns=correlation_matrix.columns,
        )

        # Based on a linear regression
        annot_fontsize = max(8 - 0.115 * n_vars, 2)
        sns.heatmap(
            correlation_matrix,
            annot=annot_matrix,
            fmt="",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.75},
            vmin=-1,
            vmax=1,
            ax=ax,
            annot_kws={"size": annot_fontsize},
        )
        ax.set_title(f"Observed correlation: {corr_name}", fontsize=11)

        # Rotate labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center", va="top")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.tick_params(axis="both", which="major", labelsize=6)
        plt.subplots_adjust(bottom=0.15)

        fig.tight_layout()

        output_path = self._create_output_dir(output_dir)
        if output_path is not None:
            filename = output_path / f"{corr_name}_heatmap.png"
            fig.savefig(filename, dpi=200)
            print(f" - Saved correlation heatmap: {filename}")

        if show:
            plt.show()

        plt.close(fig)


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


if __name__ == "__main__":
    # Testing an experimenting with this class is easier with an example,
    # rather than trying to formally test the design of output plots
    # using units tests and the like. Therefore an example is included.

    # Create sample data
    rng = np.random.default_rng(42)
    n_samples = 500

    # Generate correlated data
    mean = [0, 0]
    cov = [[1, 0.7], [0.7, 1]]
    correlated_data = rng.multivariate_normal(mean, cov, n_samples)

    df = pd.DataFrame(
        {
            "COST": correlated_data[:, 0] * 100 + 1000,
            "EFFICIENCY": correlated_data[:, 1] * 0.1 + 0.8,
            "MATERIAL": rng.choice(
                ["Steel", "Aluminum", "Titanium"], n_samples, p=[0.5, 0.3, 0.2]
            ),
        }
    )

    variables: dict[str, list[Any]] = {
        "COST": ["Normal", [1000, 100]],
        "EFFICIENCY": ["Normal", [0.8, 0.1]],
        "MATERIAL": ["Discrete", ["Steel", "Aluminum", "Titanium"]],
    }

    # Create QualityReporter
    quality_reporter = QualityReporter(df, variables)

    # 1. Plot all columns
    quality_reporter.plot_columns()

    # 2. Plot individual numeric variable
    fig, ax = quality_reporter.plot_numeric(df["COST"], "COST", variables["COST"])
    plt.show()
    plt.close(fig)

    # 3. Plot individual discrete variable
    fig, ax = quality_reporter.plot_discrete(
        df["MATERIAL"], "MATERIAL", variables["MATERIAL"]
    )
    plt.show()
    plt.close(fig)

    # 4. Correlation analysis and plotting
    correlation_matrix = pd.DataFrame(
        [[1.0, 0.7], [0.7, 1.0]],
        index=["COST", "EFFICIENCY"],
        columns=["COST", "EFFICIENCY"],
    )

    quality_reporter.print_correlation("corr1", correlation_matrix)
    quality_reporter.plot_correlation("corr1", correlation_matrix, show=True)
