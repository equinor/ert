from setuptools import setup, find_packages


import os


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", path, filename))
    return paths


extra_files = package_files("ert_gui/resources/")
logging_configuration = package_files("ert_logging/")


setup(
    name="Ensemble Reservoir Tool",
    use_scm_version={"root": ".", "write_to": "ert_shared/version.py"},
    scripts=["ert_shared/bin/ert"],
    packages=[
        "ert_data",
        "ert_logging",
        "ert_shared",
        "ert_shared.models",
        "ert_shared.plugins",
        "ert_shared.plugins.hook_specifications",
        "ert_shared.hook_implementations",
        "ert_shared.storage",
        "ert_gui",
        "ert_gui.ertwidgets",
        "ert_gui.ide",
        "ert_gui.plottery",
        "ert_gui.simulation",
        "ert_gui.tools",
        "ert_gui.ertwidgets.models",
        "ert_shared.ide.completers",
        "ert_shared.ide.keywords",
        "ert_gui.ide.wizards",
        "ert_shared.ide.keywords.data",
        "ert_shared.ide.keywords.definitions",
        "ert_gui.plottery.plots",
        "ert_gui.tools.export",
        "ert_gui.tools.help",
        "ert_gui.tools.ide",
        "ert_gui.tools.load_results",
        "ert_gui.tools.manage_cases",
        "ert_gui.tools.plot",
        "ert_gui.tools.plugins",
        "ert_gui.tools.run_analysis",
        "ert_gui.tools.workflows",
        "ert_gui.tools.plot.customize",
        "ert_gui.tools.plot.widgets",
    ],
    package_data={"ert_gui": extra_files, "ert_logging": logging_configuration},
    include_package_data=True,
    license="Open Source",
    long_description=open("README.md").read(),
    install_requires=[
        "ansicolors==1.1.8",
        "console-progressbar==1.1.2",
        "decorator",
        "flask",
        "jinja2",
        "matplotlib < 3; python_version < '3.0'",
        "matplotlib < 3.2; python_version >= '3.0'",
        "numpy",
        "pandas",
        "pluggy",
        "pyyaml",
        "qtpy",
        "scipy",
        "sqlalchemy",
    ],
    zip_safe=False,
    tests_require=["pytest", "mock"],
    test_suite="tests",
    setup_requires=["pytest-runner", "setuptools_scm"],
)
