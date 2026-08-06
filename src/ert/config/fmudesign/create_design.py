"""Module for generating design matrices that can be run by DESIGN2PARAMS
and DESIGN_KW in FMU/ERT.


A DesignMatrix is a "God-object" that contains information about all info
used to generate design matrices, including one or several Sensitivities.


"""

from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import probabilit

import semeio
from semeio.fmudesign import design_distributions as design_dist
from semeio.fmudesign._excel_to_dict import _raise_if_duplicates
from semeio.fmudesign.config_validation import validate_configuration
from semeio.fmudesign.quality_report import QualityReporter, print_corrmat
from semeio.fmudesign.utils import (
    find_max_realisations,
    map_dependencies,
    parameters_from_extern,
    printwarning,
    to_numeric_safe,
)

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence


class DesignMatrix:
    """Class for design matrix in FMU. Can contain a onebyone design
    or a full montecarlo design.

    Attributes:
        designvalues (pd.DataFrame): design matrix on standard fmu format
            contains columns 'REAL' (realization number), and if a onebyone
            design, also columns 'SENSNAME' and 'SENSCASE'
        defaultvalues (dict): default values for design
        backgroundvalues (pd.DataFrame): Used when background parameters are
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

        """
        self.designvalues: pd.DataFrame
        self.defaultvalues: dict[Hashable, Any] = {}
        self.backgroundvalues: pd.DataFrame | None = None
        self.seedvalues: list[int] | None = None
        self.verbosity: int = verbosity
        self.output_dir: Path | None = output_dir

    def reset(self) -> None:
        """Resets DesignMatrix to empty. Necessary iin case method generate
        is used several times for same instance of DesignMatrix"""
        self.designvalues = pd.DataFrame()
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

        self.designvalues["SENSNAME"] = None
        self.designvalues["SENSCASE"] = None

        for key in inputdict["sensitivities"]:
            sens = inputdict["sensitivities"][key]

            # Numer of realization (rows) to use for each sensitivity
            size = sens["numreal"] if "numreal" in sens else inputdict["repeats"]

            print(f" Generating sensitivity : {key}")

            if sens["senstype"] == "ref":
                sensitivity = SingleRealisationReference(key, verbosity=self.verbosity)
                sensitivity.generate(size=size)
                sensitivity.map_dependencies(sens.get("dependencies", {}))
                self._add_sensitivity(sensitivity)
            elif sens["senstype"] == "background":
                sensitivity = BackgroundSensitivity(key, verbosity=self.verbosity)
                sensitivity.generate(size=size)
                sensitivity.map_dependencies(sens.get("dependencies", {}))
                self._add_sensitivity(sensitivity)
            elif sens["senstype"] == "seed":
                sensitivity = SeedSensitivity(key, verbosity=self.verbosity)
                sensitivity.generate(
                    size=size,
                    seedname=sens["seedname"],
                    seedvalues=self.seedvalues,
                    parameters=sens["parameters"],
                )
                sensitivity.map_dependencies(sens.get("dependencies", {}))

                self._add_sensitivity(sensitivity)
            elif sens["senstype"] == "scenario":
                sensitivity = ScenarioSensitivity(key, verbosity=self.verbosity)
                for casekey in sens["cases"]:
                    case = sens["cases"][casekey]
                    temp_case = ScenarioSensitivityCase(casekey)
                    temp_case.generate(
                        size=size,
                        parameters=case,
                        seedvalues=self.seedvalues,
                    )
                    sensitivity.add_case(temp_case)
                    sensitivity.map_dependencies(sens.get("dependencies", {}))

                self._add_sensitivity(sensitivity)
            elif sens["senstype"] == "dist":
                sensitivity = MonteCarloSensitivity(key, verbosity=self.verbosity)
                sensitivity.generate(
                    size=size,
                    parameters=sens["parameters"],
                    seedvalues=self.seedvalues,
                    corrdict=sens["correlations"],
                    rng=self.rng,
                    correlation_iterations=inputdict.get("correlation_iterations", 0),
                )
                sensitivity.map_dependencies(sens.get("dependencies", {}))

                self._add_sensitivity(sensitivity)

            elif sens["senstype"] == "extern":
                sensitivity = ExternSensitivity(key, verbosity=self.verbosity)
                sensitivity.generate(
                    size=size,
                    filename=sens["extern_file"],
                    parameters=sens["parameters"],
                    seedvalues=self.seedvalues,
                )
                sensitivity.map_dependencies(sens.get("dependencies", {}))

                self._add_sensitivity(sensitivity)

            else:
                raise ValueError(f"Unknown sensitivity type: {sens['senstype']!r}")

            # MonteCarloSensitivity is special - it can produce debugging outputs
            is_montecarlo = isinstance(sensitivity, MonteCarloSensitivity)
            if is_montecarlo and self.verbosity > 0:
                sensitivity = cast("MonteCarloSensitivity", sensitivity)
                quality_reporter = QualityReporter(
                    df=sensitivity.sensvalues, variables=sens["parameters"]
                )

                # Print to terminal
                quality_reporter.print_numeric()
                quality_reporter.print_discrete()
                for corr_name, df_corr in sensitivity.correlation_dfs_.items():
                    quality_reporter.print_correlation(corr_name, df_corr)

            if is_montecarlo and self.verbosity > 1 and self.output_dir is not None:
                sensitivity = cast("MonteCarloSensitivity", sensitivity)
                output_dir = self.output_dir / key
                quality_reporter.plot_columns(output_dir=output_dir)

                # Correlations
                for corr_name, df_corr in sensitivity.correlation_dfs_.items():
                    # Always plot heatmaps
                    quality_reporter.plot_correlation_heatmap(
                        corr_name, df_corr, output_dir=output_dir, show=False
                    )

                    # Only plot pairgrid for small correlations
                    if len(df_corr) <= 6:
                        quality_reporter.plot_correlation(
                            corr_name, df_corr, output_dir=output_dir, show=False
                        )

        # Once all sensitivities have been added, complete the work
        if "background" in inputdict:
            self._fill_with_background_values()
        self._fill_with_defaultvalues()

        # Round columns in `self.designvalues` to desired precision
        self._set_decimals(inputdict)

        # Create REAL column (realization number)
        self.designvalues = self.designvalues.assign(REAL=lambda df: np.arange(len(df)))

        # Re-order columns
        start_cols = ["REAL", "SENSNAME", "SENSCASE", "RMS_SEED"]
        self.designvalues = self.designvalues[
            [col for col in start_cols if col in self.designvalues]
            + [col for col in self.designvalues if col not in start_cols]
        ]

        # Make all values numerical if possible
        self.designvalues = self.designvalues.map(to_numeric_safe)

    def to_xlsx(
        self,
        filename: str,
        designsheet: str = "DesignSheet01",
        defaultsheet: str = "DefaultValues",
    ) -> None:
        """Writing design matrix to excel workfbook on standard fmu format
        to be used in FMU/ERT by DESIGN2PARAMS and DESIGN_KW

        Args:
            filename (str): output filename (extension .xlsx)
            designsheet (str): name of excel sheet containing design matrix
                (optional, defaults to 'DesignSheet01')
            defaultsheet (str): name of excel sheet containing default
                values (optional, defaults to 'DefaultValues')
        """
        # Create folder for output file
        Path(filename).parent.mkdir(exist_ok=True, parents=True)

        if not filename.endswith(".xlsx"):
            filename += ".xlsx"
            print(f"Warning: Missing .xlsx suffix. Changed to: {filename}")

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

            version_info = pd.DataFrame(
                {
                    "Description": ["Created using semeio version:", "Created on:"],
                    "Value": [
                        semeio.__version__,
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
    def create_rms_seeds(seeds: list | str | None, max_reals: int) -> list | None:
        """Create RMS seems from 'seeds' argument.

        Args:
            seeds: Seed configuration. Can be:
                - None: returns None
                - "default": Generates sequential seeds 1001, 1002, 1003, ...
                - list of seeds, e.g. [1, 2, 3]
            max_reals: Maximum number of seed values to generate or load

        Examples
        --------
        >>> DesignMatrix.create_rms_seeds([1, 2, 3], max_reals=5)
        Provided number of seed values (3) in external file is lower than the maximum number of realisations (5).
         Seeds will be repeated, e.g. [1, 2, 3] => [1, 2, 3, 1, 2, ...]
        [1, 2, 3, 1, 2]
        """  # noqa: E501
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

    def background_to_excel(
        self, filename: str, backgroundsheet: str = "Background"
    ) -> None:
        """Writing background values to an Excel spreadsheet

        Args:
            filename (str): output filename (extension .xlsx)
            backgroundsheet (str): name of excel sheet
        """
        if self.backgroundvalues is None:
            raise ValueError("No background values available to write to Excel")

        xlsxwriter = pd.ExcelWriter(filename, engine="openpyxl")
        self.backgroundvalues.to_excel(
            xlsxwriter, sheet_name=backgroundsheet, index=False, header=True
        )
        xlsxwriter.close()
        print(f"Backgroundvalues written to {filename}")

    def _add_sensitivity(
        self,
        sensitivity: Sensitivity,
    ) -> None:
        """Adding a sensitivity to the design

        Args:
            sensitivity of class Scenario, MonteCarlo or Extern
        """
        existing_values = self.designvalues
        new_values = sensitivity.sensvalues
        self.designvalues = pd.concat([existing_values, new_values])

    def _fill_with_background_values(self) -> None:
        """Substituting NaNs with background values if existing.
        background values not in design are added as separate columns
        """
        if self.backgroundvalues is None:
            return

        grouped = self.designvalues.groupby(["SENSNAME", "SENSCASE"], sort=False)
        result_values = pd.DataFrame()
        for sensname, case_ in grouped:
            temp_df = case_.reset_index()
            temp_df = temp_df.fillna(self.backgroundvalues)
            for key in self.backgroundvalues.columns:
                if key not in case_:
                    temp_df[key] = self.backgroundvalues[key]
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
            existing_values = result_values.copy()
            result_values = pd.concat([existing_values, temp_df])

        result_values = result_values.drop(["index"], axis=1)
        self.designvalues = result_values

    def _fill_with_defaultvalues(self) -> None:
        """Filling NaNs with default values"""
        for key in self.designvalues.columns:
            if key in self.defaultvalues:
                self.designvalues[key] = self.designvalues[key].fillna(
                    self.defaultvalues[key]
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
        )
        mc_backgroundvalues = mc_background.sensvalues.copy()
        quality_reporter = QualityReporter(
            df=mc_backgroundvalues, variables=back_dict["parameters"]
        )

        # Print info to terminal
        if self.verbosity > 0:
            quality_reporter.print_numeric()
            quality_reporter.print_discrete()
            for corr_name, df_corr in mc_background.correlation_dfs_.items():
                quality_reporter.print_correlation(corr_name, df_corr)

        # Write plots to disk
        if self.verbosity > 0 and self.output_dir is not None:
            output_dir = self.output_dir / mc_background.sensname
            quality_reporter.plot_columns(output_dir=output_dir)

            # Correlations
            for corr_name, df_corr in mc_background.correlation_dfs_.items():
                quality_reporter.plot_correlation(
                    corr_name, df_corr, output_dir=output_dir, show=False
                )

        # Rounding of background values as specified
        if "decimals" in back_dict:
            for key in back_dict["decimals"]:
                if design_dist.is_number(mc_backgroundvalues[key].iloc[0]):
                    mc_backgroundvalues[key] = (
                        mc_backgroundvalues[key]
                        .astype(float)
                        .round(int(back_dict["decimals"][key]))
                    )
                else:
                    raise ValueError("Cannot round a string parameter")
        self.backgroundvalues = mc_backgroundvalues.copy()

    def _set_decimals(self, inputdict: dict[str, Any]) -> None:
        """Round to specified number of decimals.

        Args:
            inputdict (dictionary): input diction that might have a sub-dict
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
                    if not inputdict["decimals"].get(from_param, None):
                        continue
                    inputdict["decimals"][to_param] = inputdict["decimals"].get(
                        from_param, ""
                    )

        # Round each column
        dict_decimals = inputdict["decimals"]
        for key in self.designvalues.columns:
            if key in dict_decimals:
                if design_dist.is_number(self.designvalues[key].iloc[0]):
                    self.designvalues[key] = (
                        self.designvalues[key]
                        .astype(float)
                        .round(int(dict_decimals[key]))
                    )
                else:
                    raise ValueError(f"Cannot round a string parameter {key}")


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

    def map_dependencies(self, dependencies: dict[str, Any]) -> Sensitivity:
        """Map the dependencies, mutating the dataframe `self.sensvalues`."""
        verbose = self.verbosity > 0  # Because the function takes a boolean
        self.sensvalues: pd.DataFrame = map_dependencies(
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
            raise ValueError(msg)

        self.sensvalues = pd.DataFrame(index=range(size))
        self.sensvalues[seedname] = seedvalues[0:size]

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
            realnums (list): list of integers with realization numbers
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
    realizatons in a ScenarioSensitivityCase have identical values
    but one or more parameter has a different values from the other
    ScenarioSensitivityCase.

    Exception is the seed value and the special case where
    varying background parameters are specified. Then these are varying
    within the case.

    Attributes:
        case1 (ScenarioSensitivityCase): first case, e.g. 'low case'
        case2 (ScenarioSensitivityCase): second case, e.g. 'high case'
        sensvalues (pd.DataFrame): design values for the sensitivity, containing
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
                senscase.sensvalues["SENSNAME"] = self.sensname
                self.sensvalues = pd.concat(
                    [self.sensvalues, senscase.sensvalues], sort=True
                )
        elif senscase.sensvalues is not None and "SENSCASE" in senscase.sensvalues:
            self.case1 = senscase
            self.sensvalues = senscase.sensvalues.copy()
            self.sensvalues["SENSNAME"] = self.sensname


class ScenarioSensitivityCase(Sensitivity):
    """Each ScenarioSensitivity can contain one or
    two ScenarioSensitivityCases.

    The 1-2 cases are typically 'low' and 'high' cases for one or
    a set of  parameters, where all realisatons in
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
            seeds (str): default or None
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
            rng (numpy.random.Generator): Random number generator instance
            correlation_iterations (int): Number of permutations performed
              on samples after Iman-Conover in an attempt to match observed
              correlation to desired correlation as well as possible.
        """
        self.sensvalues = pd.DataFrame(
            columns=list(parameters.keys()), index=range(size)
        )
        self.correlation_dfs_ = {}  # Store correlation matrices (dataframes)

        if size < 0:
            raise ValueError(f"Got < 0 samples ({size=})")

        distr_by_name = {}
        for param_name, (dist_name, dist_params, _) in parameters.items():
            # Convert to a probabilit Distribution object
            distr = design_dist.to_probabilit(
                distname=dist_name, dist_parameters=dist_params
            )
            distr_by_name[param_name] = distr

        # Create a dummy NoOp node for sampling each parent distribution
        expression = probabilit.modeling.NoOp(*distr_by_name.values())

        if corrdict:
            # Create an iterator over correlation groups from the main sheet
            df_params = (
                pd.DataFrame.from_dict(
                    parameters,
                    orient="index",
                    columns=["dist_name", "dist_params", "corr_sheet"],
                )
                .reset_index()
                .rename(columns={"index": "param_name"})
                .assign(corr_sheet=lambda df: df.corr_sheet.fillna("nocorr"))
            )

            corr_groups = dict(iter(df_params.groupby("corr_sheet")))
            corr_groups.pop("nocorr", None)

            for corr_group_name, corr_group in corr_groups.items():
                corr_group_name = cast("str", corr_group_name)

                # Skip nocorr
                if corr_group_name == "nocorr":
                    continue

                # A single correlation - print warning and skip it
                if len(corr_group) == 1:
                    printwarning(corr_group_name)
                    continue

                # Read correlation matrix and convert it to a proper matrix
                df_correlations = design_dist.read_correlations(
                    excel_filename=corrdict["inputfile"], corr_sheet=corr_group_name
                )
                multivariate_parameters = df_correlations.index.tolist()
                correlations = df_correlations.to_numpy()

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
                        f"\nWarning: Correlation matrix {corr_group_name!r} "
                        "is inconsistent"
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

                corrvars = [distr_by_name[name] for name in multivariate_parameters]
                expression.correlate(*corrvars, corr_mat=df_correlations.to_numpy())
                self.correlation_dfs_[corr_group_name] = df_correlations

        # Either do ImanConover followed by Permutation, or simply ImanConover
        if correlation_iterations > 0:
            correlator = probabilit.correlation.Composite(
                iterations=correlation_iterations,
                correlation_type="pearson",
                random_state=rng,
                verbose=False,
            )
        else:
            correlator = probabilit.correlation.ImanConover()

        # Sample the dummy node - this samples every parent and populates "samples_"
        expression.sample(
            size=size, random_state=rng, method="lhs", correlator=correlator
        )

        for distr_name, distr_obj in distr_by_name.items():
            samples = distr_obj.samples_
            is_numeric = issubclass(samples.dtype.type, np.number)
            if is_numeric and not np.all(np.isfinite(distr_obj.samples_)):
                raise ValueError(
                    f"Sampling produced non-finite values in {distr_name}={distr_obj}\n"
                    "Please review the parameters in the distribution."
                )

            # Discrete distributions are handled in a special way. We map them
            # to Uniform distributions, sample in [0, 1), then map those samples
            # back to the categorical values AFTER sampling. This is so that we
            # can "induce correlations" between categorical values.
            if hasattr(distr_obj, "_values"):
                probabilities = getattr(distr_obj, "_probabilities", None)
                samples = design_dist.quantiles_to_values(
                    quantiles=samples,
                    values=distr_obj._values,
                    probabilities=probabilities,
                )

            self.sensvalues = self.sensvalues.assign(**{distr_name: samples})

        if self.sensname != "background":
            self.sensvalues["SENSNAME"] = self.sensname
            self.sensvalues["SENSCASE"] = "p10_p90"
            if "RMS_SEED" not in self.sensvalues and seedvalues:
                self.sensvalues["RMS_SEED"] = seedvalues[:size]

        null_columns = self.sensvalues.isna().any(axis=0)
        if null_columns.any():
            cols_w_null = list(null_columns.loc[lambda ser: ser].index)
            raise ValueError(f"Found NaN values in columns: {cols_w_null}")


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
            seeds (str): default or None
        """
        _raise_if_duplicates(parameters)
        self.sensvalues = pd.DataFrame(columns=parameters, index=range(size))
        extern_values = parameters_from_extern(filename)
        if size > len(extern_values):
            raise ValueError(
                f"Number of realisations {size} specified for "
                f"sensitivity {self.sensname} is larger than rows in "
                f"file {filename}"
            )
        for param in parameters:
            if param in extern_values:
                self.sensvalues[param] = list(extern_values[param][:size])
            else:
                raise ValueError(f"Parameter {param} not in external file")

        self.sensvalues["SENSNAME"] = self.sensname
        self.sensvalues["SENSCASE"] = "p10_p90"

        if seedvalues:
            self.sensvalues["RMS_SEED"] = seedvalues[:size]
