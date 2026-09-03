from pathlib import Path


def onebyone_configuration():
    return {
        "designtype": "onebyone",
        "repeats": 10,
        "distribution_seed": None,
        "seeds": "default",
        "background": None,
        "defaultvalues": {
            "RMS_SEED": 1000,
            "FAULT_POSITION": 0,
            "DC_MODEL": "base",
            "OWC1": 2650,
            "OWC2": 2750,
            "OWC3": 2850,
            "MULTZ_ILE": "0.1",
            "PARAM1": 0,
            "PARAM2": "0.1",
            "PARAM3": 0,
            "PARAM4": 0,
        },
        "sensitivities": {
            "rms_seed": {
                "seedname": "RMS_SEED",
                "senstype": "seed",
                "parameters": None,
                "dependencies": {},
            },
            "faults": {
                "cases": {
                    "east": {"FAULT_POSITION": -1},
                    "west": {"FAULT_POSITION": 1.0},
                },
                "senstype": "scenario",
                "dependencies": {},
            },
            "velmodel": {
                "cases": {"alternative": {"DC_MODEL": "alternative"}},
                "senstype": "scenario",
                "dependencies": {},
            },
            "contacts": {
                "cases": {
                    "shallow": {"OWC1": 2600, "OWC2": 2700, "OWC3": 2800},
                    "deep": {"OWC1": 2700.0, "OWC2": 2800.0, "OWC3": 2900.0},
                },
                "senstype": "scenario",
                "dependencies": {},
            },
            "multz": {
                "senstype": "dist",
                "parameters": {
                    "MULTZ_ILE": ["logunif", [0.0001, 1.0], None],
                },
                "correlations": None,
                "numreal": 20,
                "dependencies": {},
            },
        },
        "decimals": {},
    }


def full_mc_configuration(correlation_workbook: Path):
    correlation_file = str(correlation_workbook)
    return {
        "designtype": "onebyone",
        "repeats": 1,
        "distribution_seed": 555,
        "seeds": "default",
        "background": None,
        "defaultvalues": {
            "RMS_SEED": 1000,
            "HUM_MODE": "PREDICTION",
            "HUM_METHOD": "SIMPLE",
            "INJ_UNC": 0,
            "PARAM1": 0.035,
            "PARAM2": 0.05,
            "PARAM3": 0.02,
            "OWC1": 0.001,
            "OWC2": 0.1,
            "OWC3": 0.07,
            "NTG1": 0.25,
            "NTG2": 0.45,
            "NTG6": 0.8,
            "DATO": "2018-11-03",
            "FAULTSEAL": 0.01,
            "DERIVED_PARAM1": 1,
            "DERIVED_PARAM2": "a",
        },
        "sensitivities": {
            "montecarlo": {
                "senstype": "dist",
                "parameters": {
                    "PARAM1": ["triang", [0.025, 0.035, 0.045], "corr1"],
                    "PARAM2": ["norm", [0, 1, -1.0, 1.0], "corr1"],
                    "PARAM3": ["logn", [1, 10], "corr1"],
                    "OWC1": ["unif", [2500, 2550], "corr2"],
                    "OWC2": ["triang", [2430, 2470, 2500.0], "corr2"],
                    "OWC3": ["triang", [2400, 2450, 2500.0], "corr2"],
                    "DATO": [
                        "disc",
                        [
                            "2018-11-02, 2018-11-03, 2018-11-04",
                            "0.3, 0.4, 0.3",
                        ],
                        "corr3",
                    ],
                    "NTG1": ["triang", [0.75, 0.8, 1.0], "corr3"],
                    "NTG2": ["pert", [0.5, 0.6, 0.65], None],
                    "FAULTSEAL": ["logunif", [0.001, 1], None],
                },
                "correlations": {
                    "inputfile": correlation_file,
                    "sheetnames": ["corr1", "corr2", "corr3"],
                },
                "numreal": 500,
                "dependencies": {
                    "DATO": {
                        "from_values": [
                            "2018-11-02",
                            "2018-11-03",
                            "2018-11-04",
                        ],
                        "to_params": {
                            "DERIVED_PARAM1": ["1", "2", "3"],
                            "DERIVED_PARAM2": ["a", "b", "c"],
                        },
                    }
                },
            }
        },
        "decimals": {
            "PARAM1": 3,
            "PARAM2": 2,
            "PARAM3": 3,
            "OWC1": 1,
            "OWC2": 1,
            "OWC3": 1,
            "NTG1": 2,
            "NTG2": 2,
            "FAULTSEAL": 3,
        },
    }


def background_configuration(
    correlation_workbook: Path,
    external_parameters: Path,
):
    correlation_file = str(correlation_workbook)
    return {
        "designtype": "onebyone",
        "repeats": 5,
        "distribution_seed": None,
        "seeds": "default",
        "background": {
            "correlations": {
                "inputfile": correlation_file,
                "sheetnames": ["background_corr"],
            },
            "parameters": {
                "PARAM17": ["normal", [0, 1], "background_corr"],
                "PARAM18": ["uniform", [0, 1], "background_corr"],
                "PARAM19": ["triang", [1, 3, 5.0], "background_corr"],
            },
            "decimals": {"PARAM17": 2, "PARAM18": 2, "PARAM19": 2},
        },
        "defaultvalues": {
            "DEFAULT1": 0,
            "DEFAULT2": "prediction",
            "DEFAULT3": 0,
            "PARAM1": 0,
            "DC_MODEL": "base",
            "PARAM2": 0,
            "PARAM3": 0,
            "PARAM4": 0,
            "MULTZ_ILE": "0.1",
            "PARAM5": 0,
            "PARAM6": 0,
            "PARAM7": 0,
            "PARAM8": 0,
            "PARAM9": 0,
            "PARAM10": 0,
            "PARAM11": 0,
            "PARAM12": 0,
            "PARAM13": 0,
            "PARAM14": 0,
            "PARAM15": 0,
            "PARAM16": 0,
            "PARAM17": 1,
            "PARAM18": 0,
            "PARAM19": 0.5,
            "FAULT_SEAL": "base",
            "PARAM20": 0.5,
        },
        "sensitivities": {
            "background": {"senstype": "background", "dependencies": {}},
            "faults": {
                "cases": {
                    "low": {"PARAM1": -1},
                    "high": {"PARAM1": 1.0},
                },
                "senstype": "scenario",
                "dependencies": {},
            },
            "velmodel": {
                "cases": {"alternative": {"DC_MODEL": "alternative"}},
                "senstype": "scenario",
                "dependencies": {},
            },
            "contacts": {
                "cases": {
                    "shallow": {"PARAM2": -1, "PARAM3": -1, "PARAM4": -1},
                    "deep": {"PARAM2": 1.0, "PARAM3": 1.0, "PARAM4": 1.0},
                },
                "senstype": "scenario",
                "dependencies": {},
            },
            "multz": {
                "senstype": "dist",
                "parameters": {
                    "MULTZ_ILE": ["logunif", ["0.0001", 1], None],
                },
                "correlations": None,
                "numreal": 10,
                "dependencies": {},
            },
            "sens6": {
                "senstype": "dist",
                "parameters": {
                    "PARAM5": ["normal", [3, 1, 1.0, 5.0], "corr0"],
                    "PARAM6": ["uniform", [0, 1], "corr0"],
                    "PARAM7": ["triang", [1, 3, 5.0], None],
                    "FAULT_SEAL": [
                        "discrete",
                        [
                            "2018-11-02, 2018-11-03, 2018-11-04",
                            "0.3, 0.4, 0.3",
                        ],
                        None,
                    ],
                },
                "correlations": {
                    "inputfile": correlation_file,
                    "sheetnames": ["corr0"],
                },
                "numreal": 500,
                "dependencies": {},
            },
            "sens7": {
                "senstype": "dist",
                "parameters": {
                    "PARAM9": ["lognormal", [0, 1], "corr1"],
                    "PARAM10": ["uniform", [0, 1], "corr1"],
                    "PARAM11": ["triang", [0, 0.5, 1.0], "corr1"],
                    "PARAM12": ["logunif", [1, 10], "corr1"],
                },
                "correlations": {
                    "inputfile": correlation_file,
                    "sheetnames": ["corr1"],
                },
                "numreal": 500,
                "dependencies": {},
            },
            "sens8": {
                "extern_file": str(external_parameters),
                "senstype": "extern",
                "parameters": ["PARAM13", "PARAM14", "PARAM15", "PARAM16"],
                "numreal": 11,
                "dependencies": {},
            },
            "sens9": {
                "senstype": "dist",
                "parameters": {
                    "PARAM17": ["uniform_p10_p90", [0, 1], None],
                    "PARAM18": ["normal_p10_p90", [0.1, 0.9], None],
                    "PARAM19": ["triangular_p10_p90", [0.1, 0.5, 0.9], None],
                    "PARAM20": ["pert_p10_p90", [10, 50, 90.0], None],
                },
                "correlations": None,
                "numreal": 500,
                "dependencies": {},
            },
        },
        "decimals": {"PARAM9": 2, "PARAM10": 3},
    }
