"""Module for generating design matrices that can be run by DESIGN2PARAMS
and DESIGN_KW in FMU/ERT.


A DesignMatrix is a "God-object" that contains information about all info
used to generate design matrices, including one or several Sensitivities.


"""

from __future__ import annotations

import getpass
import hashlib
import logging
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import probabilit.correlation
import probabilit.modeling

from ert.shared import __version__ as ert_version
from fmudesign.exceptions import FmudesignError

from .design_config import DesignConfig
from .design_distributions import (
    DiscreteViaUniform,
    read_correlations,
    to_probabilit,
)
from .quality_report import QualityReporter, print_corrmat
from .seed_strategy import SeedStrategy
from .utils import (
    _raise_if_duplicates,
    map_dependencies,
    parameters_from_extern,
    to_numeric_safe,
)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy.typing as npt

    # (group_name, correlation_matrix, member_params)
    CorrelationGroup = tuple[str, pd.DataFrame, list[str]]
    ProbabilitNode = probabilit.modeling.Node[npt.NDArray[Any]]


def _normalize_xlsx_filename(filename: str) -> str:
    return filename if filename.endswith(".xlsx") else f"{filename}.xlsx"


def _derive_rng(base_seed: int, *keys: str) -> np.random.Generator:
    """Return a numpy Generator seeded from ``base_seed`` and ``keys``.

    Keys are ``(sensname, "param", param_name)`` for an uncorrelated parameter
    and ``(sensname, "corr", group_name)`` for a correlation group.

    Components are length-prefixed so that no two distinct key tuples can hash
    to the same payload, e.g. ``("a", "b:c")`` versus ``("a:b", "c")``.
    """
    hasher = hashlib.sha256()
    for component in (str(base_seed), *keys):
        data = component.encode("utf-8")
        hasher.update(len(data).to_bytes(4, "big"))
        hasher.update(data)
    return np.random.default_rng(int.from_bytes(hasher.digest(), "big"))


def _round_to_decimals(df: pd.DataFrame, decimals: Mapping[str, int]) -> None:
    """Round, in place, each column of `df` that has a number of decimals."""
    for key in df.columns:
        if key in decimals:
            try:
                values = df[key].astype(float)
            except (TypeError, ValueError) as err:
                raise FmudesignError(
                    f"Cannot round parameter {key!r} to {decimals[key]} decimals "
                    f"({err}). Either clear its 'decimals' cell or make all its "
                    "values numeric."
                ) from err
            df[key] = values.round(int(decimals[key]))


class DesignMatrix:
    """Class for design matrix in FMU. Can contain a one-by-one design
    or a full Monte Carlo design.

    Attributes:
        designvalues (pd.DataFrame): design matrix on standard fmu format
            contains columns 'REAL' (realization number), and if a onebyone
            design, also columns 'SENSNAME' and 'SENSCASE'
        defaultvalues (dict): default values for design
        backgroundvalues (pd.DataFrame): Used when background parameters are
            not constant. Either a set is sampled from specified distributions
            or they are read from a file.
        seedvalues: list of seed values
        verbosity: how much information to print
        output_dir: where to write debugging output and QC plots
        rng: generator seeded with 'distribution_seed'; draws all parameters
            under 'joint'
        seed_strategy: 'joint' or 'independent' Monte Carlo seeding
        base_seed: root seed that 'independent' derives one generator per
            parameter and per correlation group from. Equals
            'distribution_seed', or a draw from rng when no seed is given.
            Unused under 'joint'
    """

    designvalues: pd.DataFrame
    defaultvalues: dict[str, str | float | int | bool]
    backgroundvalues: pd.DataFrame | None
    seedvalues: list[int] | None
    verbosity: int
    output_dir: Path | None
    rng: np.random.Generator
    seed_strategy: SeedStrategy
    base_seed: int

    START_COLUMNS = ("REAL", "SENSNAME", "SENSCASE", "RMS_SEED")

    def __init__(
        self,
        config: DesignConfig,
        verbosity: int = 0,
        output_dir: Path | None = None,
    ) -> None:
        self.designvalues = pd.DataFrame()
        self.verbosity: int = verbosity
        self.output_dir: Path | None = output_dir
        self._generate(config)

    def log_inputdict(self, config: DesignConfig) -> None:
        sensitivities: dict[str, Any] = config.sensitivities
        sampled_parameters: list[dict[str, Any]] = [
            sens_values["parameters"]
            for sens_values in sensitivities.values()
            if sens_values["senstype"] == "dist"
        ]
        background_source = None
        decimals = set(config.decimals or {})
        if isinstance(background := config.background, dict):
            sampled_parameters.append(background.get("parameters", {}))
            decimals |= set(background.get("decimals", {}))
            background_source = "extern" if "extern" in background else "distributions"
        parameters = [
            param for params in sampled_parameters for param in params.items()
        ]
        correlation_groups: defaultdict[tuple[int, str], Counter[str]] = defaultdict(
            Counter
        )
        for source, source_parameters in enumerate(sampled_parameters):
            for dist_name, _, corr_sheet in source_parameters.values():
                if corr_sheet is not None:
                    correlation_groups[source, corr_sheet][dist_name] += 1
        non_parameter_columns = set(self.START_COLUMNS) - {"RMS_SEED"}
        background_params = (
            set()
            if self.backgroundvalues is None
            else set(self.backgroundvalues.columns) - non_parameter_columns
        )
        non_numeric_values = self.designvalues.drop(
            columns=list(non_parameter_columns)
        ).select_dtypes(exclude="number")
        is_text = non_numeric_values.map(lambda value: isinstance(value, str))

        senstype_to_unique_params: dict[str, set[str]] = defaultdict(set)
        params_per_sensitivity: list[tuple[str, int]] = []
        reals_per_sensitivity: list[tuple[str, int]] = []
        sensitivities_per_param: Counter[str] = Counter()
        for sensvals in sensitivities.values():
            reals = sensvals.get("numreal", config.repeats)
            if (senstype := sensvals["senstype"]) == "scenario":
                params = {
                    param for case in sensvals["cases"].values() for param in case
                }
                reals *= len(sensvals["cases"])
            else:
                params = set(sensvals.get("parameters") or {})
            derived_params = {
                param
                for dependency in sensvals.get("dependencies", {}).values()
                for param in dependency["to_params"]
            }
            senstype_to_unique_params[senstype] |= params
            params_per_sensitivity.append((senstype, len(params)))
            reals_per_sensitivity.append((senstype, reals))
            sensitivities_per_param.update(params | derived_params)
        senstype_to_param_count: dict[str, int] = {
            senstype: len(params)
            for senstype, params in senstype_to_unique_params.items()
        }

        summary_log = dedent(
            f"""\
            Fmudesign summary:
            Repeats: {config.repeats}
            Seed strategy: {config.seed_strategy}
            Correlation iterations: {config.correlation_iterations}
            Number of sensitivities: {len(sensitivities)}
            Number of sensitivities per type: {
                dict(Counter(sens["senstype"] for sens in sensitivities.values()))
            }
            Number of background parameters: {len(background_params)}
            Number of background parameters also set in sensitivities: {
                len(background_params & sensitivities_per_param.keys())
            }
            Distribution count (name, parameter count): {
                dict(Counter((dist[0], len(dist[1])) for _, dist in parameters))
            }
            Parameters per sensitivity type in designinput: {senstype_to_param_count}
            Parameters in designmatrix: {
                len(set(self.designvalues.columns) - non_parameter_columns)
            }
            Number of non-numeric parameters in designmatrix: {
                non_numeric_values.shape[1]
            }
            Number of parameters mixing text and numbers in designmatrix: {
                (is_text.any() & ~is_text.all()).sum()
            }
            Parameters in designmatrix defaultsheet: {len(self.defaultvalues)}
            Number of parameters with correlations: {
                len(
                    {
                        param
                        for param, (_, _, corr_sheet) in parameters
                        if corr_sheet is not None
                    }
                )
            }
            Distributions per correlation group: {
                [dict(group) for group in correlation_groups.values()]
            }
            Background: {background_source}
            RMS seeds: {
                "file" if isinstance(config.rms_seeds, list) else config.rms_seeds
            }
            Number of parameters with decimals: {len(decimals)}
            Distribution seed: {config.distribution_seed is not None}
            Number of dependencies: {
                sum(
                    len(sens.get("dependencies", {})) for sens in sensitivities.values()
                )
            }
            Number of parameters per sensitivity: {params_per_sensitivity}
            Number of parameters set in more than one sensitivity: {
                sum(count > 1 for count in sensitivities_per_param.values())
            }
            Number of realizations per sensitivity: {reals_per_sensitivity}
            Total number of realizations: {len(self.designvalues)}"""
        )
        logger.info(summary_log)

    def _generate(self, config: DesignConfig) -> None:
        """Adds default values and background values if they exist.
        Loops through sensitivities and adds them to designvalues.

        Args:
            config (DesignConfig): configuration for generating a DesignMatrix
        """
        self.rng = np.random.default_rng(seed=config.distribution_seed)
        self.defaultvalues = config.defaultvalues

        self.seed_strategy = config.seed_strategy
        distribution_seed = config.distribution_seed
        self.base_seed = (
            distribution_seed
            if distribution_seed is not None
            else int(self.rng.integers(2**63))
        )

        # Reading or generating rms seed values
        max_reals = max(
            [sensinfo.get("numreal", 0) for sensinfo in config.sensitivities.values()]
            + [config.repeats]
        )
        self.seedvalues = DesignMatrix.create_rms_seeds(config.rms_seeds, max_reals)

        self.add_background(
            back_dict=config.background,
            max_values=max_reals,
            correlation_iterations=config.correlation_iterations,
        )

        sensitivity: Sensitivity

        self.designvalues["SENSNAME"] = None
        self.designvalues["SENSCASE"] = None

        for key, sens in config.sensitivities.items():
            # Number of realisations (rows) to use for each sensitivity
            size = sens.get("numreal", config.repeats)

            print(f" Generating sensitivity : {key}")

            match sens["senstype"]:
                case "ref":
                    sensitivity = SingleRealisationReference(
                        key, verbosity=self.verbosity
                    )
                    sensitivity.generate(size=size)
                case "background":
                    sensitivity = BackgroundSensitivity(key, verbosity=self.verbosity)
                    sensitivity.generate(size=size)
                case "seed":
                    sensitivity = SeedSensitivity(key, verbosity=self.verbosity)
                    sensitivity.generate(
                        size=size,
                        seedname=sens["seedname"],
                        seedvalues=self.seedvalues,
                        parameters=sens["parameters"],
                    )
                case "scenario":
                    sensitivity = ScenarioSensitivity(key, verbosity=self.verbosity)
                    for casekey, case in sens["cases"].items():
                        temp_case = ScenarioSensitivityCase(casekey)
                        temp_case.generate(
                            size=size,
                            parameters=case,
                            seedvalues=self.seedvalues,
                        )
                        sensitivity.add_case(temp_case)
                case "dist":
                    sensitivity = MonteCarloSensitivity(key, verbosity=self.verbosity)
                    sensitivity.generate(
                        size=size,
                        parameters=sens["parameters"],
                        seedvalues=self.seedvalues,
                        corrdict=sens["correlations"],
                        rng=self.rng,
                        correlation_iterations=config.correlation_iterations,
                        seed_strategy=self.seed_strategy,
                        base_seed=self.base_seed,
                    )
                case "extern":
                    sensitivity = ExternSensitivity(key, verbosity=self.verbosity)
                    sensitivity.generate(
                        size=size,
                        filename=sens["extern_file"],
                        parameters=sens["parameters"],
                        seedvalues=self.seedvalues,
                    )
                case unknown:
                    raise FmudesignError(f"Unknown sensitivity type: {unknown!r}")

            sensitivity.map_dependencies(sens.get("dependencies", {}))
            self._add_sensitivity(sensitivity)

            # MonteCarloSensitivity is special - it can produce debugging outputs
            if isinstance(sensitivity, MonteCarloSensitivity) and self.verbosity > 0:
                self._report_samples(
                    sensitivity, sens["parameters"], plot=self.verbosity > 1
                )

        # Once all sensitivities have been added, complete the work
        self._fill_with_background_values()
        self._fill_with_defaultvalues()

        # Round columns in `self.designvalues` to desired precision
        self._set_decimals(config)

        # Create REAL column (realization number)
        self.designvalues = self.designvalues.assign(REAL=lambda df: np.arange(len(df)))

        # Re-order columns
        self.designvalues = self.designvalues[
            [col for col in self.START_COLUMNS if col in self.designvalues]
            + [col for col in self.designvalues if col not in self.START_COLUMNS]
        ]

        # Make all values numerical if possible
        self.designvalues = self.designvalues.map(to_numeric_safe)

        self.log_inputdict(config)

    def to_xlsx(
        self,
        filename: str,
        designsheet: str = "DesignSheet01",
        defaultsheet: str = "DefaultValues",
        input_filename: str | None = None,
    ) -> None:
        """Writing design matrix to an Excel workbook in standard FMU format
        to be used in FMU/ERT by DESIGN2PARAMS and DESIGN_KW

        Args:
            filename (str): output filename (extension .xlsx)
            designsheet (str): name of excel sheet containing design matrix
                (optional, defaults to 'DesignSheet01')
            defaultsheet (str): name of excel sheet containing default
                values (optional, defaults to 'DefaultValues')
            input_filename (str): input workbook the design was generated from,
                recorded in the metadata when running in komodo (optional)
        """
        normalized_filename = _normalize_xlsx_filename(filename)
        if normalized_filename != filename:
            filename = normalized_filename
            print(f"Warning: Missing .xlsx suffix. Changed to: {filename}")

        # Create folder for output file
        Path(filename).parent.mkdir(exist_ok=True, parents=True)

        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            self.designvalues.to_excel(
                writer, sheet_name=designsheet, index=False, header=True
            )
            # Default values
            defaults = pd.DataFrame(
                data=list(self.defaultvalues.items()),
                columns=["defaultparameters", "defaultvalue"],
            )
            defaults.to_excel(
                writer, sheet_name=defaultsheet, index=False, header=False
            )

            metadata = {
                "Created using ert version:": ert_version,
                "Created on:": datetime.now()
                .astimezone()
                .isoformat(sep=" ", timespec="seconds"),
            }
            # Only identify the user and their paths on internal komodo installations
            if komodo_release := os.environ.get("KOMODO_RELEASE"):
                metadata["Created by:"] = getpass.getuser()
                metadata["Created using komodo release:"] = komodo_release
                if input_filename is not None:
                    metadata["Created from input file:"] = str(
                        Path(input_filename).resolve()
                    )
            pd.DataFrame(
                data=list(metadata.items()), columns=["Description", "Value"]
            ).to_excel(writer, sheet_name="Metadata", index=False)

        print(
            f"Design matrix of shape {self.designvalues.shape} written to: {filename!r}"
        )

    @staticmethod
    def create_rms_seeds(
        seeds: list[int] | str | None, max_reals: int
    ) -> list[int] | None:
        """Create RMS seeds from the 'seeds' argument.

        Args:
            seeds: Seed configuration. Can be:
                - None: returns None
                - "default": Generates sequential seeds 1000, 1001, 1002, ...
                - list of seeds, e.g. [1, 2, 3]
            max_reals: Maximum number of seed values to generate or load

        Examples
        --------
        >>> DesignMatrix.create_rms_seeds([1, 2, 3], max_reals=5)
        Provided number of seed values (3) in external file is lower than the maximum number of realisations (5).
         Seeds will be repeated, e.g. [1, 2, 3] => [1, 2, 3, 1, 2, ...]
        [1, 2, 3, 1, 2]
        """  # ruff: ignore[line-too-long]
        if seeds is None:
            return None

        if seeds == "default":
            return [item + 1000 for item in range(max_reals)]

        if isinstance(seeds, list):
            if max_reals > len(seeds):
                print(
                    f"Provided number of seed values ({len(seeds)}) in external file "
                    f"is lower than the maximum number of realisations ({max_reals}).\n"
                    " Seeds will be repeated, e.g. [1, 2, 3] => [1, 2, 3, 1, 2, ...]"
                )

            return [seeds[item % len(seeds)] for item in range(max_reals)]

        # Raise if none of the cases above apply. We do this because if we did not we
        # would return None, which is a valid case in itself.
        raise FmudesignError(f"Must be None, 'default' or list: {seeds=}")

    def add_background(
        self,
        back_dict: dict[str, Any] | None,
        max_values: int,
        correlation_iterations: int = 0,
    ) -> None:
        """Adding background as specified in dictionary.
        Either from external file or from distributions in background
        dictionary

        Seeding follows ``self.seed_strategy`` / ``self.base_seed``.

        Args:
            back_dict (dict | None): how to generate background values. If set to
              `None`, ``self.backgroundvalues`` will be set to `None`.
            max_values (int): number of background values to generate
            correlation_iterations (int): Number of permutations performed
              on samples after Iman-Conover in an attempt to match observed
              correlation to desired correlation as well as possible.
        """
        if back_dict is None:
            self.backgroundvalues = None
        elif isinstance(back_dict, dict):
            if "extern" in back_dict:
                print(f"Reading background values from: {back_dict['extern']}")
                self.backgroundvalues = parameters_from_extern(back_dict["extern"])
            elif "parameters" in back_dict:
                print("Generating background values from distributions.")
                self._add_dist_background(
                    back_dict=back_dict,
                    size=max_values,
                    correlation_iterations=correlation_iterations,
                )
            else:
                raise FmudesignError(
                    "Neither keys 'extern' or 'parameters' were set in background "
                    "dictionary - of which one is required. If no background values "
                    "are to be used, set background to 'None'."
                )
        else:
            raise FmudesignError(
                "Background value in DesignMatrix config must be 'None' or of type "
                "'dict'."
            )

    def _add_sensitivity(
        self,
        sensitivity: Sensitivity,
    ) -> None:
        """Adding a sensitivity to the design

        Args:
            sensitivity (Sensitivity): sensitivity with values to append
        """
        self.designvalues = pd.concat([self.designvalues, sensitivity.sensvalues])

    def _fill_with_background_values(self) -> None:
        """Substituting NaNs with background values if existing.
        background values not in design are added as separate columns
        """
        if self.backgroundvalues is None:
            return

        grouped = self.designvalues.groupby(["SENSNAME", "SENSCASE"], sort=False)
        result_values = pd.DataFrame()
        for sensname, case_ in grouped:
            temp_df = case_.reset_index(drop=True)
            temp_df = temp_df.fillna(self.backgroundvalues)
            for key in self.backgroundvalues.columns:
                if key not in case_:
                    temp_df[key] = self.backgroundvalues[key]
                    if len(temp_df) > len(self.backgroundvalues):
                        raise FmudesignError(
                            "Provided number of background values "
                            f"{len(self.backgroundvalues)} is smaller than number"
                            f" of realisations for sensitivity {sensname}"
                        )
                elif len(temp_df) > len(self.backgroundvalues):
                    print(
                        "Provided number of background values "
                        f"({len(self.backgroundvalues)}) is smaller than number"
                        f" of realisations for sensitivity {sensname}"
                        f" and parameter {key}. "
                        "Will be filled with default values."
                    )
            result_values = pd.concat([result_values, temp_df])

        self.designvalues = result_values

    def _fill_with_defaultvalues(self) -> None:
        """Filling NaNs with default values"""
        for key in self.designvalues.columns:
            if key in self.defaultvalues:
                self.designvalues[key] = self.designvalues[key].fillna(
                    self.defaultvalues[key]
                )
            elif key not in self.START_COLUMNS:
                raise FmudesignError(f"No defaultvalues given for parameter {key} ")

    def _add_dist_background(
        self,
        back_dict: dict[str, Any],
        size: int,
        correlation_iterations: int,
    ) -> None:
        """Drawing background values from distributions
        specified in dictionary

        Args:
            back_dict (dict): parameters and distributions
            size (int): Number of samples to generate
            correlation_iterations (int): Number of permutations performed
              on samples after Iman-Conover in an attempt to match observed
              correlation to desired correlation as well as possible.
        """

        mc_background = MonteCarloSensitivity("background")
        mc_background.generate(
            size=size,
            parameters=back_dict["parameters"],
            seedvalues=None,
            corrdict=back_dict["correlations"],
            rng=self.rng,
            correlation_iterations=correlation_iterations,
            seed_strategy=self.seed_strategy,
            base_seed=self.base_seed,
        )
        if self.verbosity > 0:
            self._report_samples(mc_background, back_dict["parameters"], plot=True)

        background_values = mc_background.sensvalues
        _round_to_decimals(background_values, back_dict.get("decimals", {}))
        self.backgroundvalues = background_values

    def _report_samples(
        self,
        sensitivity: MonteCarloSensitivity,
        variables: dict[str, Any],
        *,
        plot: bool,
    ) -> None:
        """Print statistics of the sampled values and, if `plot`, save plots
        of them to `output_dir`.
        """
        quality_reporter = QualityReporter(
            df=sensitivity.sensvalues, variables=variables
        )
        quality_reporter.print_numeric()
        quality_reporter.print_discrete()
        for corr_name, df_corr in sensitivity.correlation_dfs_.items():
            quality_reporter.print_correlation(corr_name, df_corr)

        if not plot or self.output_dir is None:
            return

        output_dir = self.output_dir / sensitivity.sensname
        quality_reporter.plot_columns(output_dir=output_dir)
        for corr_name, df_corr in sensitivity.correlation_dfs_.items():
            quality_reporter.plot_correlation_heatmap(
                corr_name, df_corr, output_dir=output_dir
            )
            # Only plot pairgrid for small correlation groups
            if len(df_corr) <= 6:
                quality_reporter.plot_correlation(
                    corr_name, df_corr, output_dir=output_dir
                )

    def _set_decimals(self, config: DesignConfig) -> None:
        """Round to specified number of decimals.

        Args:
            config (DesignConfig): configuration whose 'decimals' maps
                parameter names to their number of decimals
        """
        # No decimal information => Nothing to do.
        if not config.decimals:
            return

        dict_decimals = config.decimals.copy()

        # If there are dependencies (derived params) that are copies,
        # like TO := copy(FROM), then the new TO column must be rounded too.
        for sensdict in config.sensitivities.values():
            if not sensdict["dependencies"]:
                continue
            for from_param, from_dict in sensdict["dependencies"].items():
                for to_param, to_values in from_dict["to_params"].items():
                    if from_param in dict_decimals and not to_values:
                        dict_decimals[to_param] = dict_decimals[from_param]

        _round_to_decimals(self.designvalues, dict_decimals)


class Sensitivity:
    sensvalues: pd.DataFrame

    def __init__(self, sensname: str, verbosity: int = 0) -> None:
        """
        Args:
            sensname (str): Name of sensitivity. Defines SENSNAME in design matrix.
            verbosity (int): How much information to print. Non-negative integer.
        """
        self.sensname: str = sensname
        self.verbosity: int = verbosity

    def map_dependencies(self, dependencies: dict[str, Any]) -> None:
        """Map the dependencies, replacing the dataframe `self.sensvalues`."""
        self.sensvalues = map_dependencies(
            self.sensvalues, dependencies=dependencies, verbose=self.verbosity > 0
        )


class SeedSensitivity(Sensitivity):
    """
    A seed sensitivity is normally the reference for one by one sensitivities,
    which all other sensitivities are compared to. All parameters will be at
    their default values. Only the RMS_SEED will be varying.

    It contains a list of seeds to be repeated for each sensitivity
    The parameter name is hardcoded to RMS_SEED
    It will be assigned the sensname 'p10_p90' which will be written to
    the SENSCASE column in the output.

    Attributes:
        sensname (str): name of sensitivity
        sensvalues (pd.DataFrame):  design values for the sensitivity

    """

    def generate(
        self,
        size: int,
        seedname: str,
        seedvalues: Sequence[int] | None,
        parameters: dict[str, Any] | None,
    ) -> None:
        """Generates parameter values for a seed sensitivity

        Args:
            size (int): number of rows to generate
            seedname (str): name of seed parameter to add
            seedvalues (list): list of integer seedvalues
            parameters (dict): parameter names and
                distributions or values.
        """
        if seedvalues is None:
            msg = (
                "Seed values must be set when running sensitivity type 'seed'. "
                f"Got seed: {seedvalues}"
            )
            raise FmudesignError(msg)

        self.sensvalues = pd.DataFrame(index=range(size))
        self.sensvalues[seedname] = seedvalues[0:size]

        if parameters is not None:
            for key in parameters:
                dist_name = parameters[key][0].lower()
                constant = parameters[key][1]
                if dist_name != "const":
                    raise FmudesignError(
                        'A sensitivity of type "seed" can only have '
                        "additional parameters where dist_name is "
                        f'"const". Check sensitivity {self.sensname}"'
                    )
                self.sensvalues[key] = constant

        self.sensvalues["SENSNAME"] = self.sensname
        self.sensvalues["SENSCASE"] = "p10_p90"


class SingleRealisationReference(Sensitivity):
    """
    The class is used in set-ups where one wants a single realisation
    containing only default values as a reference, but the realisation
    itself is not included in a sensitivity.
    Typically used when RMS_SEED is not a parameter.
    SENSCASE will be set to 'ref' in design matrix, to flag that it should be
    excluded as a sensitivity in the plot.

    Attributes:
        sensname (str): name of sensitivity
        sensvalues (pd.DataFrame):  design values for the sensitivity

    """

    def generate(
        self,
        size: int,
    ) -> None:
        """Generates realisation number only

        Args:
            size (int): number of rows to generate
        """
        self.sensvalues = pd.DataFrame(index=range(size))
        self.sensvalues["SENSNAME"] = self.sensname
        self.sensvalues["SENSCASE"] = "ref"


class BackgroundSensitivity(Sensitivity):
    """
    The class is used in set-ups where one sensitivities
    are run on top of varying background parameters.
    Typically used when RMS_SEED is not a parameter, so the reference
    for tornadoplots will be the realisations with all parameters
    at their default values except the background parameters.
    SENSCASE will be set to 'p10_p90' in design matrix.

    Attributes:
        sensname (str): name of sensitivity
        sensvalues (pd.DataFrame):  design values for the sensitivity

    """

    def generate(self, size: int) -> None:
        """Generates realisation number only

        Args:
            size (int): number of rows to generate
        """
        self.sensvalues = pd.DataFrame(index=range(size))
        self.sensvalues["SENSNAME"] = self.sensname
        self.sensvalues["SENSCASE"] = "p10_p90"


class ScenarioSensitivity(Sensitivity):
    """Each design can contain one or several single sensitivities of type
    Seed, MonteCarlo or Scenario.
    Each ScenarioSensitivity can contain 1-2 ScenarioSensitivityCases.

    The ScenarioSensitivity class is used for sensitivities where all
    realisations in a ScenarioSensitivityCase have identical values
    but one or more parameter has a different values from the other
    ScenarioSensitivityCase.

    Exception is the seed value and the special case where
    varying background parameters are specified. Then these are varying
    within the case.

    Attributes:
        sensvalues (pd.DataFrame): design values for the sensitivity, containing
           1-2 cases
    """

    def add_case(self, senscase: ScenarioSensitivityCase) -> None:
        """
        Adds a ScenarioSensitivityCase instance
        to a ScenarioSensitivity object.

        Args:
            senscase (ScenarioSensitivityCase):
                Equals SENSCASE in design matrix.
        """
        case_values = senscase.sensvalues.assign(SENSNAME=self.sensname)
        if hasattr(self, "sensvalues"):
            case_values = pd.concat([self.sensvalues, case_values], sort=True)
        self.sensvalues = case_values


class ScenarioSensitivityCase(Sensitivity):
    """Each ScenarioSensitivity can contain one or
    two ScenarioSensitivityCases.

    The 1-2 cases are typically 'low' and 'high' cases for one or
    a set of parameters, where all realisations in
    the case have identical values except the seed value
    and in special cases specified background values which may
    vary within the case.

    One or two ScenarioSensitivityCase instances can be added to each
    ScenarioSensitivity object.

    Attributes:
        sensname (str): name of the sensitivity case,
            equals SENSCASE in design matrix.
        sensvalues (pd.DataFrame): parameters and values
            for the sensitivity with realisation numbers as index.

    """

    def generate(
        self,
        size: int,
        parameters: dict[str, Any],
        seedvalues: Sequence[int] | None,
    ) -> None:
        """Generate sensvalues for the ScenarioSensitivityCase

        Args:
            size (int): number of rows to generate
            parameters (dict):
                dictionary with parameter names and values
            seedvalues (list): list of integer seed values or None
        """

        self.sensvalues = pd.DataFrame(
            columns=list(parameters.keys()), index=range(size)
        )
        for key, value in parameters.items():
            self.sensvalues[key] = value
        self.sensvalues["SENSCASE"] = self.sensname

        if seedvalues:
            self.sensvalues["RMS_SEED"] = seedvalues[:size]


class MonteCarloSensitivity(Sensitivity):
    """
    For a MonteCarloSensitivity one or several parameters
    are drawn from specified distributions with or without correlations.
    A MonteCarloSensitivity can only contain
    one case, where the name SENSCASE is automatically set to 'p10_p90' in the
    design matrix to flag that p10_p90 should be calculated in TornadoPlot.

    Attributes:
        sensname (str):  name for the sensitivity.
            Equals SENSNAME in design matrix.
        sensvalues (pd.DataFrame):  parameters and values for the sensitivity
            with realisation numbers as index.
    """

    def generate(
        self,
        *,
        size: int,
        parameters: dict[str, Any],
        seedvalues: Sequence[int] | None,
        corrdict: dict[str, Any] | None,
        rng: np.random.Generator,
        correlation_iterations: int = 0,
        seed_strategy: SeedStrategy = SeedStrategy.JOINT,
        base_seed: int | None = None,
    ) -> None:
        """Generates parameter values by drawing from defined distributions.

        Args:
            size (int): number of rows to generate
            parameters (dict): dictionary of parameters and distributions
            seedvalues (list): list of integer seed values or None
            corrdict (dict): Configuration for correlated parameters. Contains:
                - 'inputfile': Name of Excel file with correlation matrices
                - 'sheetnames': List of sheet names, where each sheet contains a
                correlation matrix. If None, parameters are treated as uncorrelated.
            rng (numpy.random.Generator): Random number generator instance.
              Draws all values under 'joint'. Unused under 'independent'.
            correlation_iterations (int): Number of permutations performed
              on samples after Iman-Conover in an attempt to match observed
              correlation to desired correlation as well as possible.
            seed_strategy (SeedStrategy): How to seed the sampling.
                - 'joint' (default): all parameters are drawn in a single Latin
                  Hypercube Sampling call. Adding, removing or reordering a
                  parameter reshuffles every parameter.
                - 'independent': each uncorrelated parameter and each correlation
                  group is seeded separately from ``base_seed``, so that editing
                  one leaves the others bit-identical. This means independently
                  keyed random streams, not zero empirical correlation: unrelated
                  parameters still show incidental correlation of order
                  1/sqrt(size), exactly as they do under 'joint'.
                  Values are stable only while ``size``, ``base_seed``, the
                  sensitivity name, and the parameter's own name, distribution
                  and correlation group membership are unchanged.
            base_seed (int | None): Root seed that 'independent' derives one
              generator per parameter and per correlation group from. Required
              for 'independent', unused for 'joint'.
        """
        self.sensvalues = pd.DataFrame(
            columns=list(parameters.keys()), index=range(size)
        )
        self.correlation_dfs_: dict[str, pd.DataFrame] = {}  # correlation matrices

        if size < 0:
            raise FmudesignError(f"Got < 0 samples ({size=})")

        distr_by_name: dict[str, ProbabilitNode] = {}
        for param_name, (dist_name, dist_params, _) in parameters.items():
            try:
                distr_by_name[param_name] = to_probabilit(
                    distname=dist_name, dist_parameters=dist_params
                )
            except ValueError as err:
                raise FmudesignError(
                    f"Parameter {param_name!r} in sensitivity {self.sensname!r}: {err}"
                ) from err

        # Read and validate the correlation groups once, up front. Both seed
        # strategies consume the same groups; only the seeding differs.
        corr_groups = self._load_correlation_groups(
            parameters,
            corrdict,
            size=size,
            correlation_iterations=correlation_iterations,
        )

        if seed_strategy == SeedStrategy.JOINT:
            self._sample_joint(
                size=size,
                distr_by_name=distr_by_name,
                corr_groups=corr_groups,
                correlation_iterations=correlation_iterations,
                rng=rng,
            )
        elif seed_strategy == SeedStrategy.INDEPENDENT:
            if base_seed is None:
                raise FmudesignError(
                    "'base_seed' is required when seed_strategy='independent'"
                )
            self._sample_independent(
                size=size,
                distr_by_name=distr_by_name,
                corr_groups=corr_groups,
                correlation_iterations=correlation_iterations,
                base_seed=base_seed,
            )
        else:
            raise FmudesignError(
                f"'seed_strategy' must be one of {[s.value for s in SeedStrategy]}, "
                f"got: {seed_strategy!r}"
            )

        for distr_name, distr_obj in distr_by_name.items():
            samples = distr_obj.samples_
            if isinstance(distr_obj, DiscreteViaUniform):
                samples = distr_obj.to_values(samples)

            self.sensvalues = self.sensvalues.assign(**{distr_name: samples})

        if self.sensname != "background":
            self.sensvalues["SENSNAME"] = self.sensname
            self.sensvalues["SENSCASE"] = "p10_p90"
            if "RMS_SEED" not in self.sensvalues and seedvalues:
                self.sensvalues["RMS_SEED"] = seedvalues[:size]

        null_columns = self.sensvalues.isna().any(axis=0)
        if null_columns.any():
            cols_w_null = list(null_columns.loc[lambda ser: ser].index)
            raise FmudesignError(f"Found NaN values in columns: {cols_w_null}")

    def _load_correlation_groups(
        self,
        parameters: dict[str, Any],
        corrdict: dict[str, Any] | None,
        *,
        size: int,
        correlation_iterations: int,
    ) -> list[CorrelationGroup]:
        """Return ``(group_name, df_correlations, member_params)`` per correlation
        group, reading and validating each matrix.

        Single-member groups are skipped (with a warning) and treated as
        uncorrelated. Populates ``self.correlation_dfs_`` as a side effect.
        Shared by both the 'joint' and 'independent' seed strategies.

        Raises:
            ValueError: if a parameter is a member of more than one correlation
                group, since only one of the requested correlations could then
                be honoured.
            ValueError: if ``size`` is less than 2, since a correlation group
                needs at least 2 realizations to induce any correlation.
            ValueError: if ``correlation_iterations`` is 0 and a group has at
                least as many parameters as ``size``, since Iman-Conover needs
                more realizations than correlated parameters.
        """
        if not corrdict:
            return []

        group_sizes = Counter(
            corr_sheet
            for _, _, corr_sheet in parameters.values()
            if pd.notna(corr_sheet)
        )

        loaded: list[CorrelationGroup] = []
        group_of_param: dict[str, str] = {}
        # The order affects seeded sampling: sheet indices first, then names
        for corr_group_name in sorted(
            group_sizes, key=lambda name: (isinstance(name, str), name)
        ):
            if group_sizes[corr_group_name] == 1:
                print(
                    "Warning: Only one parameter uses correlation sheet "
                    f"{corr_group_name!r}, so it is sampled without correlation."
                )
                continue

            # The Excel sheet only fills in the lower triangle, which
            # read_correlations mirrors into a full symmetric matrix.
            df_correlations = read_correlations(
                excel_filename=corrdict["inputfile"], corr_sheet=corr_group_name
            )
            multivariate_parameters = df_correlations.index.tolist()
            correlations = df_correlations.to_numpy()

            # Each group is sampled as one unit, so a parameter in two groups
            # would get the correlations of whichever group is sampled last.
            for name in multivariate_parameters:
                if name in group_of_param:
                    raise FmudesignError(
                        f"Parameter {name!r} is part of several correlation "
                        f"groups: {group_of_param[name]!r} and "
                        f"{corr_group_name!r}. A parameter may only appear in "
                        "one correlation matrix."
                    )
                group_of_param[name] = corr_group_name

            if size < 2:
                raise FmudesignError(
                    f"Correlation group {corr_group_name!r} in sensitivity "
                    f"{self.sensname!r} needs at least 2 realizations (numreal, "
                    "or repeats if numreal is not set) to correlate its "
                    f"parameters, got {size}."
                )

            num_params = len(multivariate_parameters)
            if correlation_iterations == 0 and size <= num_params:
                raise FmudesignError(
                    f"Correlation group {corr_group_name!r} in sensitivity "
                    f"{self.sensname!r} has {num_params} parameters, but only "
                    f"{size} realizations. Inducing correlations needs more "
                    "realizations than correlated parameters. Use at least "
                    f"{num_params + 1} realizations (numreal, or repeats if "
                    "numreal is not set), or fewer parameters in the correlation "
                    "group."
                )

            if self.verbosity == 0:
                print(
                    f"Sampling {len(multivariate_parameters)} parameters",
                    f"in correlation group {corr_group_name!r}",
                )
            else:
                print(
                    f"Sampling {len(multivariate_parameters)} parameters",
                    f"in correlation group {corr_group_name!r}: "
                    f"{multivariate_parameters}",
                )

            # Get the nearest correlation matrix
            nearest = probabilit.correlation.nearest_correlation_matrix(
                correlations, weights=None, eps=1e-6, verbose=False
            )
            if not np.allclose(correlations, nearest):
                print(
                    f"\nWarning: Correlation matrix {corr_group_name!r} is inconsistent"
                )
                print("Requirements:")
                print("  - All diagonal elements must be 1")
                print("  - All elements must be between -1 and 1")
                print("  - The matrix must be positive semi-definite")
                print("\nInput correlation matrix:")
                print_corrmat(df_correlations)
                df_correlations = pd.DataFrame(
                    nearest,
                    index=df_correlations.index,
                    columns=df_correlations.columns,
                )
                print("\nAdjusted to nearest consistent correlation matrix:")
                print_corrmat(df_correlations)

            self.correlation_dfs_[corr_group_name] = df_correlations
            loaded.append((corr_group_name, df_correlations, multivariate_parameters))

        return loaded

    @staticmethod
    def _make_correlator(
        correlation_iterations: int, rng: np.random.Generator
    ) -> probabilit.correlation.Correlator:
        """Return the correlator used to induce the requested correlations.

        ``correlation_iterations=0`` gives plain Iman-Conover. A positive number
        adds that many rounds of random row swaps on top, keeping only the swaps
        that move the observed correlation closer to the target. The result is
        therefore never further off than Iman-Conover alone, and usually closer.
        """
        if correlation_iterations > 0:
            return probabilit.correlation.Composite(
                iterations=correlation_iterations,
                correlation_type="pearson",
                random_state=rng,
                verbose=False,
            )
        return probabilit.correlation.ImanConover()

    def _sample_joint(
        self,
        *,
        size: int,
        distr_by_name: dict[str, ProbabilitNode],
        corr_groups: list[CorrelationGroup],
        correlation_iterations: int,
        rng: np.random.Generator,
    ) -> None:
        """Draw all parameters in a single LHS call sharing one RNG, so the
        sample of every parameter depends on the full parameter set.
        """
        # Create a dummy NoOp node for sampling each parent distribution
        expression = probabilit.modeling.NoOp(*distr_by_name.values())

        for _name, df_correlations, member_params in corr_groups:
            corrvars = [distr_by_name[name] for name in member_params]
            expression.correlate(*corrvars, corr_mat=df_correlations.to_numpy())

        correlator = self._make_correlator(correlation_iterations, rng)

        # Sample the dummy node. This samples every parent distribution and
        # stores the draws on each distribution object as 'samples_'.
        try:
            expression.sample(
                size=size, random_state=rng, method="lhs", correlator=correlator
            )
        except ValueError as err:
            raise FmudesignError(
                f"Sampling sensitivity {self.sensname!r} failed: {err}"
            ) from err

    def _sample_independent(
        self,
        *,
        size: int,
        distr_by_name: dict[str, ProbabilitNode],
        corr_groups: list[CorrelationGroup],
        correlation_iterations: int,
        base_seed: int,
    ) -> None:
        """Sample each correlation group and each uncorrelated parameter as a
        separate unit, seeded from ``base_seed`` and a stable key. Adding,
        removing or reordering a parameter therefore leaves the other
        parameters unchanged (given a fixed seed and sample size).
        """
        grouped_params: set[str] = set()

        # Each correlation group is one independently seeded unit.
        for group_name, df_correlations, member_params in corr_groups:
            distrs = [distr_by_name[name] for name in member_params]
            expression = probabilit.modeling.NoOp(*distrs)
            expression.correlate(*distrs, corr_mat=df_correlations.to_numpy())
            unit_rng = _derive_rng(base_seed, self.sensname, "corr", group_name)
            correlator = self._make_correlator(correlation_iterations, unit_rng)
            try:
                expression.sample(
                    size=size,
                    random_state=unit_rng,
                    method="lhs",
                    correlator=correlator,
                )
            except ValueError as err:
                raise FmudesignError(
                    f"Sampling correlation group {group_name!r} in sensitivity "
                    f"{self.sensname!r} failed: {err}"
                ) from err
            grouped_params.update(member_params)

        # Each remaining (uncorrelated) parameter is its own independent unit.
        for param_name, distr in distr_by_name.items():
            if param_name in grouped_params:
                continue
            unit_rng = _derive_rng(base_seed, self.sensname, "param", param_name)
            try:
                distr.sample(size=size, random_state=unit_rng, method="lhs")
            except ValueError as err:
                raise FmudesignError(
                    f"Sampling parameter {param_name!r} in sensitivity "
                    f"{self.sensname!r} failed: {err}"
                ) from err


class ExternSensitivity(Sensitivity):
    """
    Used when reading parameter values from a file
    Assumed to be used with monte carlo type sensitivities and
    will hence write 'p10_p90' as SENSCASE in output designmatrix

    Attributes:
        sensname (str): Name of sensitivity.
            Defines SENSNAME in design matrix
        sensvalues (pd.DataFrame):  design values for the sensitivity

    """

    def generate(
        self,
        size: int,
        filename: str,
        parameters: list[str],
        seedvalues: Sequence[int] | None,
    ) -> None:
        """Reads parameter values for a monte carlo sensitivity
        from file

        Args:
            size (int): number of samples to generate
            filename (str): file to read values from
            parameters (list): list with parameter names
            seedvalues (list): list of integer seed values or None
        """
        try:
            _raise_if_duplicates(parameters)
        except ValueError as e:
            raise FmudesignError(f"Found parameters with duplicate names.\n{e}") from e
        self.sensvalues = pd.DataFrame(columns=parameters, index=range(size))
        extern_values = parameters_from_extern(filename)
        if size > len(extern_values):
            raise FmudesignError(
                f"Number of realisations {size} specified for "
                f"sensitivity {self.sensname} is larger than rows in "
                f"file {filename}"
            )
        for param in parameters:
            if param in extern_values:
                self.sensvalues[param] = list(extern_values[param][:size])
            else:
                raise FmudesignError(f"Parameter {param} not in external file")

        self.sensvalues["SENSNAME"] = self.sensname
        self.sensvalues["SENSCASE"] = "p10_p90"

        if seedvalues:
            self.sensvalues["RMS_SEED"] = seedvalues[:size]
