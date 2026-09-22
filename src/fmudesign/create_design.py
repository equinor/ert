"""Module for generating design matrices that can be run by DESIGN2PARAMS
and DESIGN_KW in FMU/ERT.


A DesignMatrix is a "God-object" that contains information about all info
used to generate design matrices, including one or several Sensitivities.


"""

from __future__ import annotations

import copy
import hashlib
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import polars as pl
import probabilit.correlation
import probabilit.modeling

from ert.shared import __version__ as ert_version

from .config_validation import SeedStrategy, validate_configuration
from .design_distributions import (
    DiscreteViaUniform,
    is_number,
    read_correlations,
    to_probabilit,
)
from .quality_report import QualityReporter, print_corrmat
from .utils import (
    _has_value,
    _raise_if_duplicates,
    concat_design_frames,
    fill_parameter_nulls,
    find_max_realisations,
    map_dependencies,
    numeric_parameter_series,
    parameter_series,
    parameters_from_extern,
    printwarning,
)

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    import numpy.typing as npt

    # (group_name, correlation_matrix, member_params)
    CorrelationGroup = tuple[str, pl.DataFrame, list[str]]
    ProbabilitNode = probabilit.modeling.Node[npt.NDArray[Any]]


def _normalize_xlsx_filename(filename: str) -> str:
    return filename if filename.endswith(".xlsx") else f"{filename}.xlsx"


def _round_parameter(column: pl.Series, decimals: int) -> pl.Series:
    values = (
        np.asarray(column.to_list(), dtype=float)
        if column.dtype in {pl.String, pl.Object}
        else column.cast(pl.Float64).to_numpy()
    )
    return pl.Series(column.name, values.round(decimals), nan_to_null=True)


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


class DesignMatrix:
    """Class for design matrix in FMU. Can contain a one-by-one design
    or a full Monte Carlo design.

    Attributes:
        designvalues (pl.DataFrame): design matrix on standard fmu format
            contains columns 'REAL' (realization number), and if a onebyone
            design, also columns 'SENSNAME' and 'SENSCASE'
        defaultvalues (dict): default values for design
        backgroundvalues (pl.DataFrame): Used when background parameters are
            not constant. Either a set is sampled from specified distributions
            or they are read from a file.
    """

    def __init__(self, verbosity: int = 0, output_dir: Path | None = None) -> None:
        """
        Placeholders for:
        designvalues: dataframe with parameters that varies
        defaultvalues: dictionary of default/base case values
        backgroundvalues: dataframe with background parameters
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
        self.designvalues: pl.DataFrame
        self.defaultvalues: dict[Hashable, Any] = {}
        self.backgroundvalues: pl.DataFrame | None = None
        self.seedvalues: list[int] | None = None
        self.verbosity: int = verbosity
        self.output_dir: Path | None = output_dir
        self.rng: np.random.Generator
        self.seed_strategy: SeedStrategy
        self.base_seed: int

    def reset(self) -> None:
        """Resets DesignMatrix to empty. Necessary in case method generate
        is used several times for same instance of DesignMatrix
        """
        self.designvalues = pl.DataFrame(
            schema={"SENSNAME": pl.String, "SENSCASE": pl.String}
        )
        self.defaultvalues = {}
        self.backgroundvalues = None
        self.seedvalues = None

    def generate(self, inputdict: dict[str, Any]) -> None:
        """Generating design matrix from input dictionary in specific
        format. Adding default values and background values if existing.
        Looping through sensitivities and adding them to designvalues.

        Args:
            inputdict (dict): input parameters for design
        """
        inputdict = validate_configuration(inputdict, verbosity=self.verbosity)

        self.reset()  # Emptying if regenerating matrix
        self.rng = np.random.default_rng(seed=inputdict.get("distribution_seed"))
        self.defaultvalues = inputdict["defaultvalues"]

        self.seed_strategy = inputdict["seed_strategy"]
        distribution_seed = inputdict.get("distribution_seed")
        self.base_seed = (
            distribution_seed
            if distribution_seed is not None
            else int(self.rng.integers(2**63))
        )

        # Reading or generating rms seed values
        max_reals = find_max_realisations(inputdict)
        self.seedvalues = DesignMatrix.create_rms_seeds(inputdict["seeds"], max_reals)

        # If background values used - read or generate
        if "background" in inputdict:
            self.add_background(
                back_dict=inputdict["background"],
                max_values=max_reals,
                correlation_iterations=inputdict.get("correlation_iterations", 0),
            )

        sensitivity: Sensitivity

        for key, sens in inputdict["sensitivities"].items():
            # Number of realisations (rows) to use for each sensitivity
            size = sens.get("numreal", inputdict["repeats"])

            print(f" Generating sensitivity : {key}")

            match sens["senstype"]:
                case "ref":
                    sensitivity = SingleRealisationReference(
                        key, verbosity=self.verbosity
                    )
                    sensitivity.generate(size=size)
                    sensitivity.map_dependencies(sens.get("dependencies", {}))
                    self._add_sensitivity(sensitivity)
                case "background":
                    sensitivity = BackgroundSensitivity(key, verbosity=self.verbosity)
                    sensitivity.generate(size=size)
                    sensitivity.map_dependencies(sens.get("dependencies", {}))
                    self._add_sensitivity(sensitivity)
                case "seed":
                    sensitivity = SeedSensitivity(key, verbosity=self.verbosity)
                    sensitivity.generate(
                        size=size,
                        seedname=sens["seedname"],
                        seedvalues=self.seedvalues,
                        parameters=sens["parameters"],
                    )
                    sensitivity.map_dependencies(sens.get("dependencies", {}))

                    self._add_sensitivity(sensitivity)
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
                        sensitivity.map_dependencies(sens.get("dependencies", {}))

                    self._add_sensitivity(sensitivity)
                case "dist":
                    sensitivity = MonteCarloSensitivity(key, verbosity=self.verbosity)
                    sensitivity.generate(
                        size=size,
                        parameters=sens["parameters"],
                        seedvalues=self.seedvalues,
                        corrdict=sens["correlations"],
                        rng=self.rng,
                        correlation_iterations=inputdict.get(
                            "correlation_iterations", 0
                        ),
                        seed_strategy=self.seed_strategy,
                        base_seed=self.base_seed,
                    )
                    sensitivity.map_dependencies(sens.get("dependencies", {}))

                    self._add_sensitivity(sensitivity)

                case "extern":
                    sensitivity = ExternSensitivity(key, verbosity=self.verbosity)
                    sensitivity.generate(
                        size=size,
                        filename=sens["extern_file"],
                        parameters=sens["parameters"],
                        seedvalues=self.seedvalues,
                    )
                    sensitivity.map_dependencies(sens.get("dependencies", {}))

                    self._add_sensitivity(sensitivity)

                case unknown:
                    raise ValueError(f"Unknown sensitivity type: {unknown!r}")

            # MonteCarloSensitivity is special - it can produce debugging outputs
            is_montecarlo = isinstance(sensitivity, MonteCarloSensitivity)
            if is_montecarlo and self.verbosity > 0:
                sensitivity = cast("MonteCarloSensitivity", sensitivity)
                quality_reporter = QualityReporter(
                    df=sensitivity.sensvalues.to_pandas(), variables=sens["parameters"]
                )

                # Print to terminal
                quality_reporter.print_numeric()
                quality_reporter.print_discrete()
                for corr_name, df_corr in sensitivity.correlation_dfs_.items():
                    quality_reporter.print_correlation(
                        corr_name, df_corr.to_pandas().set_axis(df_corr.columns)
                    )

            if is_montecarlo and self.verbosity > 1 and self.output_dir is not None:
                sensitivity = cast("MonteCarloSensitivity", sensitivity)
                output_dir = self.output_dir / key
                quality_reporter.plot_columns(output_dir=output_dir)

                # Correlations
                for corr_name, df_corr in sensitivity.correlation_dfs_.items():
                    report_corr = df_corr.to_pandas().set_axis(df_corr.columns)
                    # Always plot heatmaps
                    quality_reporter.plot_correlation_heatmap(
                        corr_name, report_corr, output_dir=output_dir, show=False
                    )

                    # Only plot pairgrid for small correlations
                    if len(df_corr) <= 6:
                        quality_reporter.plot_correlation(
                            corr_name, report_corr, output_dir=output_dir, show=False
                        )

        # Once all sensitivities have been added, complete the work
        if "background" in inputdict:
            self._fill_with_background_values()
        self._fill_with_defaultvalues()

        # Round columns in `self.designvalues` to desired precision
        self._set_decimals(inputdict)

        # Create REAL column (realization number)
        self.designvalues = self.designvalues.drop("REAL", strict=False).with_row_index(
            "REAL"
        )

        # Re-order columns
        start_cols = ["REAL", "SENSNAME", "SENSCASE", "RMS_SEED"]
        self.designvalues = self.designvalues.select(
            [col for col in start_cols if col in self.designvalues]
            + [col for col in self.designvalues.columns if col not in start_cols]
        )

        # Make all values numerical if possible
        self.designvalues = self.designvalues.with_columns(
            numeric_parameter_series(column) for column in self.designvalues
        )

    def to_xlsx(
        self,
        filename: str,
        designsheet: str = "DesignSheet01",
        defaultsheet: str = "DefaultValues",
    ) -> None:
        """Writing design matrix to an Excel workbook in standard FMU format
        to be used in FMU/ERT by DESIGN2PARAMS and DESIGN_KW

        Args:
            filename (str): output filename (extension .xlsx)
            designsheet (str): name of excel sheet containing design matrix
                (optional, defaults to 'DesignSheet01')
            defaultsheet (str): name of excel sheet containing default
                values (optional, defaults to 'DefaultValues')
        """
        normalized_filename = _normalize_xlsx_filename(filename)
        if normalized_filename != filename:
            filename = normalized_filename
            print(f"Warning: Missing .xlsx suffix. Changed to: {filename}")

        # Create folder for output file
        Path(filename).parent.mkdir(exist_ok=True, parents=True)

        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            self.designvalues.to_pandas().to_excel(
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

            version_info = pd.DataFrame(
                {
                    "Description": ["Created using ert version:", "Created on:"],
                    "Value": [
                        ert_version,
                        datetime.now()
                        .astimezone()
                        .isoformat(sep=" ", timespec="seconds"),
                    ],
                }
            )
            version_info.to_excel(writer, sheet_name="Metadata", index=False)

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
        raise ValueError(f"Must be None, 'default' or list: {seeds=}")

    def add_background(
        self,
        back_dict: dict[str, Any] | None,
        max_values: int,
        correlation_iterations: int = 0,
    ) -> None:
        """Adding background as specified in dictionary.
        Either from external file or from distributions in background
        dictionary

        Seeding follows ``self.seed_strategy`` / ``self.base_seed``, which are
        set by :meth:`generate`.

        Args:
            back_dict (dict): how to generate background values
            max_values (int): number of background values to generate
            correlation_iterations (int): Number of permutations performed
              on samples after Iman-Conover in an attempt to match observed
              correlation to desired correlation as well as possible.
        """
        if back_dict is None:
            self.backgroundvalues = None
        elif "extern" in back_dict:
            print(f"Reading background values from: {back_dict['extern']}")
            self.backgroundvalues = parameters_from_extern(back_dict["extern"])
        elif "parameters" in back_dict:
            print("Generating background values from distributions.")
            self._add_dist_background(
                back_dict=back_dict,
                size=max_values,
                correlation_iterations=correlation_iterations,
            )

    def _add_sensitivity(
        self,
        sensitivity: Sensitivity,
    ) -> None:
        """Adding a sensitivity to the design

        Args:
            sensitivity of class Scenario, MonteCarlo or Extern
        """
        self.designvalues = concat_design_frames(
            [self.designvalues, sensitivity.sensvalues]
        )

    def _fill_with_background_values(self) -> None:
        """Substituting NaNs with background values if existing.
        background values not in design are added as separate columns
        """
        if self.backgroundvalues is None:
            return

        grouped = self.designvalues.partition_by(
            ["SENSNAME", "SENSCASE"], maintain_order=True
        )
        result_values = []
        for case_ in grouped:
            sensname = case_.select("SENSNAME", "SENSCASE").row(0)
            temp_df = case_
            for key in self.backgroundvalues.columns:
                if key not in case_:
                    if len(temp_df) > len(self.backgroundvalues):
                        raise ValueError(
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
                values = self.backgroundvalues[key].head(len(case_))
                if len(values) < len(case_):
                    values = values.extend_constant(None, len(case_) - len(values))
                if key in case_:
                    values = fill_parameter_nulls(case_[key], values)
                temp_df = temp_df.with_columns(values)
            result_values.append(temp_df)

        if result_values:
            self.designvalues = concat_design_frames(result_values)

    def _fill_with_defaultvalues(self) -> None:
        """Filling NaNs with default values"""
        for key in self.designvalues.columns:
            if key in self.defaultvalues:
                self.designvalues = self.designvalues.with_columns(
                    fill_parameter_nulls(
                        self.designvalues[key],
                        parameter_series(key, [self.defaultvalues[key]]),
                    )
                )
            elif key not in {"REAL", "SENSNAME", "SENSCASE", "RMS_SEED"}:
                raise LookupError(f"No defaultvalues given for parameter {key} ")

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
        mc_backgroundvalues = mc_background.sensvalues.clone()

        # Print info to terminal
        if self.verbosity > 0:
            quality_reporter = QualityReporter(
                df=mc_backgroundvalues.to_pandas(), variables=back_dict["parameters"]
            )
            quality_reporter.print_numeric()
            quality_reporter.print_discrete()
            for corr_name, df_corr in mc_background.correlation_dfs_.items():
                quality_reporter.print_correlation(
                    corr_name, df_corr.to_pandas().set_axis(df_corr.columns)
                )

        # Write plots to disk
        if self.verbosity > 0 and self.output_dir is not None:
            output_dir = self.output_dir / mc_background.sensname
            quality_reporter.plot_columns(output_dir=output_dir)

            # Correlations
            for corr_name, df_corr in mc_background.correlation_dfs_.items():
                quality_reporter.plot_correlation(
                    corr_name,
                    df_corr.to_pandas().set_axis(df_corr.columns),
                    output_dir=output_dir,
                    show=False,
                )

        # Rounding of background values as specified
        if "decimals" in back_dict:
            for key in back_dict["decimals"]:
                if is_number(mc_backgroundvalues[key][0]):
                    mc_backgroundvalues = mc_backgroundvalues.with_columns(
                        _round_parameter(
                            mc_backgroundvalues[key], int(back_dict["decimals"][key])
                        )
                    )
                else:
                    raise ValueError("Cannot round a string parameter")
        self.backgroundvalues = mc_backgroundvalues

    def _set_decimals(self, inputdict: dict[str, Any]) -> None:
        """Round to specified number of decimals.

        Args:
            inputdict (dictionary): input dictionary that might have a sub-dict
                                    with key "decimals". This sub-dict has
                                    (key, value)s are (param, decimals)
        """
        inputdict = copy.deepcopy(inputdict)

        # No decimal information => Nothing to do.
        if not inputdict.get("decimals", {}):
            return

        # If there are dependencies (derived params) that are copies,
        # like TO := copy(FROM), then the new TO column must be rounded too.
        for sensdict in inputdict["sensitivities"].values():
            if not sensdict["dependencies"]:
                continue
            for from_param, from_dict in sensdict["dependencies"].items():
                for to_param in from_dict["to_params"]:
                    if from_param not in inputdict["decimals"]:
                        continue
                    inputdict["decimals"][to_param] = inputdict["decimals"][from_param]

        # Round each column
        dict_decimals = inputdict["decimals"]
        for key in self.designvalues.columns:
            if key in dict_decimals:
                if is_number(self.designvalues[key][0]):
                    self.designvalues = self.designvalues.with_columns(
                        _round_parameter(
                            self.designvalues[key], int(dict_decimals[key])
                        )
                    )
                else:
                    raise ValueError(f"Cannot round a string parameter {key}")


class Sensitivity:
    sensvalues: pl.DataFrame

    def __init__(self, sensname: str, verbosity: int = 0) -> None:
        """
        Args:
            sensname (str): Name of sensitivity. Defines SENSNAME in design matrix.
            verbosity (int): How much information to print. Non-negative integer.
        """
        self.sensname: str = sensname
        self.verbosity: int = verbosity

    def map_dependencies(self, dependencies: dict[str, Any]) -> Sensitivity:
        """Map the dependencies, mutating the dataframe `self.sensvalues`."""
        verbose = self.verbosity > 0  # Because the function takes a boolean
        self.sensvalues = map_dependencies(
            self.sensvalues, dependencies=dependencies, verbose=verbose
        )
        return self


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
        sensvalues (pl.DataFrame):  design values for the sensitivity

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
            raise ValueError(msg)

        self.sensvalues = pl.DataFrame(
            {
                seedname: seedvalues[:size],
                "SENSNAME": [self.sensname] * size,
                "SENSCASE": ["p10_p90"] * size,
            }
        )

        if parameters is not None:
            for key in parameters:
                dist_name = parameters[key][0].lower()
                constant = parameters[key][1]
                if dist_name != "const":
                    raise ValueError(
                        'A sensitivity of type "seed" can only have '
                        "additional parameters where dist_name is "
                        f'"const". Check sensitivity {self.sensname}"'
                    )
                self.sensvalues = self.sensvalues.with_columns(
                    parameter_series(
                        key,
                        constant if isinstance(constant, list) else [constant] * size,
                    )
                )


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
        sensvalues (pl.DataFrame):  design values for the sensitivity

    """

    def generate(
        self,
        size: int,
    ) -> None:
        """Generates realisation number only

        Args:
            realnums (list): list of integers with realization numbers
        """
        self.sensvalues = pl.DataFrame(
            {"SENSNAME": [self.sensname] * size, "SENSCASE": ["ref"] * size}
        )


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
        sensvalues (pl.DataFrame):  design values for the sensitivity

    """

    def generate(self, size: int) -> None:
        """Generates realisation number only

        Args:
            size (int): number of rows to generate
        """
        self.sensvalues = pl.DataFrame(
            {"SENSNAME": [self.sensname] * size, "SENSCASE": ["p10_p90"] * size}
        )


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
        case1 (ScenarioSensitivityCase): first case, e.g. 'low case'
        case2 (ScenarioSensitivityCase): second case, e.g. 'high case'
        sensvalues (pl.DataFrame): design values for the sensitivity, containing
           1-2 cases
    """

    case1: ScenarioSensitivityCase | None = None
    case2: ScenarioSensitivityCase | None = None

    def add_case(self, senscase: ScenarioSensitivityCase) -> None:
        """
        Adds a ScenarioSensitivityCase instance
        to a ScenarioSensitivity object.

        Args:
            senscase (ScenarioSensitivityCase):
                Equals SENSCASE in design matrix.
        """
        if self.case1 is not None:  # Case 1 has been read, this is case2
            if senscase.sensvalues is not None and "SENSCASE" in senscase.sensvalues:
                self.case2 = senscase
                senscase.sensvalues = senscase.sensvalues.with_columns(
                    pl.lit(self.sensname).alias("SENSNAME")
                )
                self.sensvalues = concat_design_frames(
                    [self.sensvalues, senscase.sensvalues]
                )
                self.sensvalues = self.sensvalues.select(
                    sorted(self.sensvalues.columns)
                )
        elif senscase.sensvalues is not None and "SENSCASE" in senscase.sensvalues:
            self.case1 = senscase
            self.sensvalues = senscase.sensvalues.with_columns(
                pl.lit(self.sensname).alias("SENSNAME")
            )


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
        sensvalues (pl.DataFrame): parameters and values
            for the sensitivity in realization order.

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
            seeds (str): default or None
        """

        self.sensvalues = pl.DataFrame(
            [parameter_series(key, [value] * size) for key, value in parameters.items()]
            + [pl.Series("SENSCASE", [self.sensname] * size, dtype=pl.String)]
        )

        if seedvalues:
            self.sensvalues = self.sensvalues.with_columns(
                pl.Series("RMS_SEED", seedvalues[:size])
            )


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
        sensvalues (pl.DataFrame): parameters and values for the sensitivity
            in realization order.
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
            values (list): a list of seed values or None
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
        self.correlation_dfs_: dict[str, pl.DataFrame] = {}  # correlation matrices

        if size < 0:
            raise ValueError(f"Got < 0 samples ({size=})")

        distr_by_name: dict[str, ProbabilitNode] = {}
        for param_name, (dist_name, dist_params, _) in parameters.items():
            distr_by_name[param_name] = to_probabilit(
                distname=dist_name, dist_parameters=dist_params
            )

        # Read and validate the correlation groups once, up front. Both seed
        # strategies consume the same groups; only the seeding differs.
        corr_groups = self._load_correlation_groups(parameters, corrdict)

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
                raise ValueError(
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
            raise ValueError(
                f"'seed_strategy' must be one of {[s.value for s in SeedStrategy]}, "
                f"got: {seed_strategy!r}"
            )

        sampled_columns = []
        for distr_name, distr_obj in distr_by_name.items():
            samples = distr_obj.samples_
            is_numeric = issubclass(samples.dtype.type, np.number)
            if is_numeric and not np.all(np.isfinite(samples)):
                raise ValueError(
                    f"Sampling produced non-finite values in {distr_name}={distr_obj}\n"
                    "Please review the parameters in the distribution."
                )

            if isinstance(distr_obj, DiscreteViaUniform):
                samples = distr_obj.to_values(samples)

            sampled_columns.append(pl.Series(distr_name, samples, nan_to_null=True))

        self.sensvalues = pl.DataFrame(sampled_columns)
        if self.sensname != "background":
            self.sensvalues = self.sensvalues.with_columns(
                pl.lit(self.sensname).alias("SENSNAME"),
                pl.lit("p10_p90").alias("SENSCASE"),
            )
            if "RMS_SEED" not in self.sensvalues and seedvalues:
                self.sensvalues = self.sensvalues.with_columns(
                    pl.Series("RMS_SEED", seedvalues[:size])
                )

        cols_w_null = [column.name for column in self.sensvalues if column.null_count()]
        if cols_w_null:
            raise ValueError(f"Found NaN values in columns: {cols_w_null}")

    def _load_correlation_groups(
        self,
        parameters: dict[str, Any],
        corrdict: dict[str, Any] | None,
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
        """
        if not corrdict:
            return []

        groups: dict[str, list[str]] = {}
        for param_name, (_, _, corr_sheet) in parameters.items():
            if (
                corr_sheet is not None
                and _has_value(corr_sheet)
                and corr_sheet != "nocorr"
            ):
                groups.setdefault(corr_sheet, []).append(param_name)

        loaded: list[CorrelationGroup] = []
        group_of_param: dict[str, str] = {}
        for corr_group_name in sorted(groups):
            # A single correlation - print warning and skip it
            if len(groups[corr_group_name]) == 1:
                printwarning(corr_group_name)
                continue

            # The Excel sheet only fills in the lower triangle, which
            # read_correlations mirrors into a full symmetric matrix.
            df_correlations = read_correlations(
                excel_filename=corrdict["inputfile"], corr_sheet=corr_group_name
            )
            multivariate_parameters = df_correlations.columns
            correlations = df_correlations.to_numpy()

            # Each group is sampled as one unit, so a parameter in two groups
            # would get the correlations of whichever group is sampled last.
            for name in multivariate_parameters:
                if name in group_of_param:
                    raise ValueError(
                        f"Parameter {name!r} is part of several correlation "
                        f"groups: {group_of_param[name]!r} and "
                        f"{corr_group_name!r}. A parameter may only appear in "
                        "one correlation matrix."
                    )
                group_of_param[name] = corr_group_name

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
                print_corrmat(
                    df_correlations.to_pandas().set_axis(df_correlations.columns)
                )
                df_correlations = pl.DataFrame(
                    nearest,
                    schema=df_correlations.columns,
                    orient="row",
                )
                print("\nAdjusted to nearest consistent correlation matrix:")
                print_corrmat(
                    df_correlations.to_pandas().set_axis(df_correlations.columns)
                )

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
        expression.sample(
            size=size, random_state=rng, method="lhs", correlator=correlator
        )

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
            expression.sample(
                size=size, random_state=unit_rng, method="lhs", correlator=correlator
            )
            grouped_params.update(member_params)

        # Each remaining (uncorrelated) parameter is its own independent unit.
        for param_name, distr in distr_by_name.items():
            if param_name in grouped_params:
                continue
            unit_rng = _derive_rng(base_seed, self.sensname, "param", param_name)
            distr.sample(size=size, random_state=unit_rng, method="lhs")


class ExternSensitivity(Sensitivity):
    """
    Used when reading parameter values from a file
    Assumed to be used with monte carlo type sensitivities and
    will hence write 'p10_p90' as SENSCASE in output designmatrix

    Attributes:
        sensname (str): Name of sensitivity.
            Defines SENSNAME in design matrix
        sensvalues (pl.DataFrame):  design values for the sensitivity

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
            seeds (str): default or None
        """
        _raise_if_duplicates(parameters)
        extern_values = parameters_from_extern(filename)
        if size > len(extern_values):
            raise ValueError(
                f"Number of realisations {size} specified for "
                f"sensitivity {self.sensname} is larger than rows in "
                f"file {filename}"
            )
        for param in parameters:
            if param not in extern_values:
                raise ValueError(f"Parameter {param} not in external file")

        self.sensvalues = (
            extern_values.head(size)
            .select(parameters)
            .with_columns(
                pl.lit(self.sensname).alias("SENSNAME"),
                pl.lit("p10_p90").alias("SENSCASE"),
            )
        )

        if seedvalues:
            self.sensvalues = self.sensvalues.with_columns(
                pl.Series("RMS_SEED", seedvalues[:size])
            )
