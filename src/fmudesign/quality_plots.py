import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as sp
import seaborn as sns
from probabilit.modeling import Distribution

from .design_distributions import to_probabilit

if TYPE_CHECKING:
    from matplotlib.patches import Rectangle

COLORS = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])


def _plot_title(var_name: str, var_description: list[Any]) -> str:
    dist_name, dist_parameters = var_description[:2]
    arguments = ", ".join(
        f"dist_param{i}={value}" for i, value in enumerate(dist_parameters, start=1)
    )
    return f"{var_name}\n{dist_name}~({arguments})"


class QualityPlotter:
    """This class is responsible for plotting a dataframe with samples and
    saving the figures to disk.
    """

    def __init__(self, df: pd.DataFrame, variables: dict[str, list[Any]]) -> None:
        """Initialize QualityPlotter with dataframe and variable descriptions.

        Args:
            df: DataFrame containing the samples
            variables: Dictionary mapping variable names to their
                distribution descriptions
                      e.g., {"COST": ["normal", [0.0, 1.0]]}
        """
        self.df: pd.DataFrame = df.loc[:, list(variables.keys())]
        self.variables = variables
        assert not self.df.empty

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

        ax.set_title(_plot_title(var_name, var_description), fontsize=7)

        # Add plot of expected PDF
        dist_name, dist_parameters = var_description[:2]
        dist = to_probabilit(dist_name, dist_parameters)

        if isinstance(dist, Distribution):
            x = np.linspace(series.min(), series.max(), 1000)
            pdf = dist.to_scipy().pdf(x)
            ax.plot(x, pdf, color="red", lw=2, ls="--", label="expected PDF")
        else:
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

        value_counts = series.value_counts().sort_index()
        proportions = value_counts / value_counts.sum()
        sns.barplot(x=value_counts.index, y=proportions.to_numpy(), ax=ax)

        ax.set_title(_plot_title(var_name, var_description), fontsize=7)
        ax.set_xlabel(var_name)
        ax.set_ylabel("Proportion")

        # Add percentage labels on bars
        for count, p in zip(value_counts, ax.patches, strict=False):
            rect = cast("Rectangle", p)
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

    def plot_correlation(
        self,
        corr_name: str,
        df_corr: pd.DataFrame,
        *,
        output_dir: Path | None = None,
    ) -> None:
        """Plot correlation group of variables.

        Args:
            corr_name: Name of the correlation group
            df_corr: DataFrame containing the correlation structure to plot
            output_dir: Optional directory path to save plots
        """
        # Short circuit this case, as there is nothing to do
        if output_dir is None:
            return

        def corrfunc(
            x: npt.NDArray[np.float64],
            y: npt.NDArray[np.float64],
            **kwargs: Any,
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

        def add_grid(*args: Any, **kwargs: Any) -> None:
            ax = plt.gca()
            ax.grid(visible=True, ls="--", alpha=0.5)

        df = self.df[df_corr.columns].select_dtypes(include="number")

        # Only plot if two or more variables remain
        if df.shape[1] <= 1:
            return

        pairgrid = sns.PairGrid(df)

        # Parameters with identical values have no joint density to estimate
        pairgrid.map_upper(sns.kdeplot, warn_singular=False)
        pairgrid.map_upper(add_grid)
        bins = max(int(math.sqrt(len(df))), 30)
        pairgrid.map_diag(sns.histplot, bins=bins, kde=True)

        pairgrid.map_lower(sns.scatterplot, s=10, alpha=0.6)
        pairgrid.map_lower(corrfunc)

        # Add rugplots to diagonal plots
        for i, var in enumerate(df.columns):
            pairgrid.diag_axes[i].scatter(
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

        plt.close(pairgrid.fig)

    def plot_correlation_heatmap(
        self,
        corr_name: str,
        df_corr: pd.DataFrame,
        *,
        output_dir: Path | None = None,
    ) -> None:
        """Plot correlation heatmap of group of variables.

        Args:
            corr_name: Name of the correlation group
            df_corr: DataFrame containing the correlation structure to plot
            output_dir: Optional directory path to save plots
        """
        # Short circuit this case, as there is nothing to do
        if output_dir is None:
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

        fig.tight_layout()

        output_path = self._create_output_dir(output_dir)
        if output_path is not None:
            filename = output_path / f"{corr_name}_heatmap.png"
            fig.savefig(filename, dpi=200)
            print(f" - Saved correlation heatmap: {filename}")

        plt.close(fig)
